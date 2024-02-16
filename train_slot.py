import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SeqTaggingClsDataset
from model import SeqTagger
from utils import Vocab
from seqeval.metrics import classification_report
from seqeval.scheme import IOB2

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]


def main(args):
    # TODO: implement main function
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, SeqTaggingClsDataset] = {
        split: SeqTaggingClsDataset(split_data, vocab, tag2idx, args.max_len)
        for split, split_data in data.items()
    }

    # TODO: crecate DataLoader for train / dev datasets

    dataloaderTrain = DataLoader(dataset= datasets[TRAIN].data, batch_size=args.batch_size, shuffle=True, collate_fn=datasets[TRAIN].collate_fn)
    dataloaderEval = DataLoader(dataset= datasets[DEV].data, batch_size=args.batch_size, shuffle=True, collate_fn=datasets[DEV].collate_fn)
    dataloaderSeqeval = DataLoader(dataset= datasets[DEV].data, collate_fn=datasets[DEV].collate_fn)
    
    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    # TODO: init model and move model to target device(cpu / gpu)
    device = torch.device(args.device)
    model = SeqTagger(embeddings=embeddings,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            bidirectional=args.bidirectional,
            num_class=datasets[TRAIN].num_classes
            )
    model.to(device)
    # TODO: init optimizer
    lr = args.lr
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)
    # early stopping parameters
    last_loss = 100
    patience = 5 # max tolerance
    trigger = 0 # trigger of early stopping
    # train and validation
    model.train()
    epoch_pbar = args.num_epoch
    for epoch in range(1, epoch_pbar+1):
        print("Epoch:", epoch)
        # TODO: Training loop - iterate over train dataloader and update model weights
        glob_loss,glob_acc=0.0,0.0
        tokens_acc=0.0
        count,batch_times=0.0,0.0 # count = iteration
        data_train=tqdm(dataloaderTrain)
        for _,data in enumerate(data_train):
            acc, acc_token = 0.0, 0.0
            count+=1
            input=data["tokens"].to(args.device)
            label=data["tags"].to(args.device)
            # print(input.shape[0])
            input_data_len=input.shape[1]
            input_batch=input.shape[0]
            label=label.view(-1)
            label=label.to(torch.int64)
            pred=model(input)
            pred=pred.view(-1,9)
            loss = loss_fn(pred,label)
            glob_loss+=loss
            
            pred_re = torch.reshape(pred, (input_batch,input_data_len,9))
            label_re = torch.reshape(label, (input_batch,input_data_len))
            index_pred_re = pred_re.argmax(dim=2)
            for m in range(input_batch):
                n=(index_pred_re[m]==label_re[m])
                acc_token += n.type(torch.float).sum().item()
                if (n.type(torch.float).sum().item()==input_data_len):
                    acc+=1

            glob_acc+=acc
            tokens_acc+=acc_token
            batch_times+=input_batch

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        glob_loss /= count
        glob_acc /= batch_times
        tokens_acc /= (batch_times * args.max_len)
        print("loss: ",glob_loss.item(), "  joint_acc: ", glob_acc, "   token_acc: ", tokens_acc)
        # TODO: Evaluation loop - calculate accuracy and save model weights
        with torch.no_grad():
            total_loss,total_acc=0.0,0.0
            tokens_acc_eval=0.0
            count_,batch_times_eval=0.0,0.0 # count = iteration
            data_eval=tqdm(dataloaderEval)
            for _,data in enumerate(data_eval):
                acc = 0.0
                count_+=1
                input=data["tokens"].to(args.device)
                label=data["tags"].to(args.device)

                input_data_len=input.shape[1]
                input_batch=input.shape[0]

                pred=model(input)
                # label=torch.flatten(label)
                label=label.view(-1)
                # pred = torch.flatten(pred, start_dim=0, end_dim=1)
                pred=pred.view(-1,9)
                loss = loss_fn(pred,label)
                total_loss+=loss

                pred_re = torch.reshape(pred, (input_batch,input_data_len,9))
                label_re = torch.reshape(label, (input_batch,input_data_len))
                index_pred_re = pred_re.argmax(dim=2)

                for m in range(len(input)):
                    n=(index_pred_re[m]==label_re[m])
                    acc_token += n.type(torch.float).sum().item()
                    if (n.type(torch.float).sum().item()==input_batch):
                        acc+=1
                total_acc+=acc
                tokens_acc_eval+=acc_token
                batch_times_eval+=input_batch

            total_loss /= count_
            total_acc /= batch_times_eval
            tokens_acc_eval /= (batch_times * args.max_len)
            print("loss_eval: ",total_loss.item(), "    joint_acc_eval: ", total_acc, " token_acc_eval: ", tokens_acc_eval)

            if total_loss >= last_loss:
                trigger += 1
                print("trigger:", trigger)
                if trigger >= patience:
                    print('Early stopping!')
                    # torch.save(model.state_dict(), args.ckpt_dir / 'model_state_dict.pt')
                    break
            else:
                trigger = 0
                print("trigger:", trigger)
            last_loss =  total_loss
    print("Finish all epoch")
    # torch.save(model.state_dict(), args.ckpt_dir / 'model_state_dict.pt')
    # seqeval
    model = SeqTagger(embeddings=embeddings,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            bidirectional=args.bidirectional,
            num_class=datasets[DEV].num_classes
            )
    model.load_state_dict(torch.load(args.ckpt_dir / 'best.pt'), strict=False)
    model.to(device)
    model.eval()
    with torch.no_grad():
        tags = []
        label = []
        data_seqeval=tqdm(dataloaderSeqeval)
        for _, data in enumerate(data_seqeval):
            tag=data["label"]
            label.append(tag[0])
            input=data["tokens"].to(device)
            pred=model(input)
            index_pred = pred.argmax(dim=2)
            index_pred = index_pred.cpu().numpy()
            tags_pred=''
            temp=[]
            for m in range(data["len_tokens"][0]):
                
                tags_pred = str(datasets[DEV].idx2label(index_pred[0][m]))
                temp.append(tags_pred)
            tags.append(temp)
    print("\nclassification report:\n")
    print(classification_report(y_true=label, y_pred=tags,scheme=IOB2, mode='strict'))




def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/slot/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/slot/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-4)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )
    parser.add_argument("--num_epoch", type=int, default=100)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)