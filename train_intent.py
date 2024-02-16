from cgi import test
import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict
from sklearn.model_selection import learning_curve

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm

from dataset import SeqClsDataset
from utils import Vocab

from model import SeqClassifier

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, SeqClsDataset] = {
        split: SeqClsDataset(split_data, vocab, intent2idx, args.max_len)
        for split, split_data in data.items()
    }

    # TODO: crecate DataLoader for train / dev datasets

    dataloaderTrain = DataLoader(dataset= datasets[TRAIN].data, batch_size=args.batch_size, shuffle=True, collate_fn=datasets[TRAIN].collate_fn)

    dataloaderEval = DataLoader(dataset= datasets[DEV].data, batch_size=args.batch_size, shuffle=True, collate_fn=datasets[DEV].collate_fn)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    # TODO: init model and move model to target device(cpu / gpu)
    device = torch.device(args.device)
    model = SeqClassifier(embeddings=embeddings,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            bidirectional=args.bidirectional,
            num_class=datasets[TRAIN].num_classes
            )
    # model.load_state_dict(torch.load('model_state_dict.pt'))
    model.to(device)
    # TODO: init optimizer
    lr = args.lr
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # early stopping parameters
    last_loss = 100
    patience = 4 # max tolerance
    trigger = 0 # trigger of early stopping
    # train and validation
    model.train()
    epoch_pbar = args.num_epoch
    for epoch in range(1, epoch_pbar+1):
        print("Epoch:", epoch)
        # TODO: Training loop - iterate over train dataloader and update model weights
        glob_loss,glob_acc=0.0,0.0
        count,batch_times=0.0,0.0 # count = iteration
        data_train=tqdm(dataloaderTrain)
        for _,data in enumerate(data_train):
            count+=1
            input=data["text"].to(device)
            label=data["intent"].to(device)
            pred=model(input)
            loss = loss_fn(pred,label)
            glob_loss+=loss
            acc=(pred.argmax(dim=1)==label).type(torch.float).sum().item()
            glob_acc+=acc
            batch_times+=len(input)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        glob_loss /= count
        glob_acc /= batch_times
        print("loss: ",glob_loss.item(), "acc: ", glob_acc)
        # TODO: Evaluation loop - calculate accuracy and save model weights
        # model.eval()
        with torch.no_grad():
            total_loss,total_acc=0.0, 0.0
            count_, batch_times_val=0.0, 0.0 # count_ = iteration
            data_eval=tqdm(dataloaderEval)
            for _, data in enumerate(data_eval):
                count_+=1
                input=data["text"].to(device)
                label=data["intent"].to(device)
                pred=model(input)
                loss_=loss_fn(pred,label)
                total_loss+=loss_
                acc_=(pred.argmax(dim=1)==label).type(torch.float).sum().item()
                total_acc+=acc_
                batch_times_val+=len(input)
            total_loss /= count_
            total_acc /= batch_times_val
            print("loss_eval: ",total_loss.item(), "acc_eval: ", total_acc)

            if total_loss >= last_loss:
                trigger += 1
                print("trigger:", trigger)
                if trigger >= patience:
                    print('Early stopping!')
                    torch.save(model.state_dict(), 'ckpt/intent/model_state_dict.pt')
                    break
            else:
                trigger = 0
                print("trigger:", trigger)
            last_loss =  total_loss
    print("Finish all epoch")
    torch.save(model.state_dict(), 'ckpt/intent/model_state_dict.pt')

    # TODO: Inference on test set



def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/intent/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/intent/",
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
