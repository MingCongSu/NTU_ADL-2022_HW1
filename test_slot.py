import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SeqTaggingClsDataset
from model import SeqTagger
from utils import Vocab


def main(args):
    # TODO: implement main function
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())

    data = json.loads(args.test_file.read_text())
    #data = {"test": json.loads(args.test_file.read_text())}
    dataset = SeqTaggingClsDataset(data, vocab, tag2idx, args.max_len)
    # TODO: crecate DataLoader for test dataset
    dataloader = DataLoader(dataset= dataset, collate_fn=dataset.collate_fn)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    device = args.device
    model = SeqTagger(
        embeddings,
        args.hidden_size,
        args.num_layers,
        args.dropout,
        args.bidirectional,
        dataset.num_classes,
    )
    ckpt = torch.load(args.ckpt_dir / 'best.pt')
    # load weights into model
    model.load_state_dict(ckpt, strict=False)
    model.to(device)
    # TODO: predict dataset
    model.eval()
    with torch.no_grad():
        count = 0.0
        tags = ['tags']
        id = ['id']
        data_test=tqdm(dataloader)
        for _, data in enumerate(data_test):
            count+=1
            input=data["tokens"].to(device)
            pred=model(input)
            # print(pred)
            index_pred = pred.argmax(dim=2)
            index_pred = index_pred.cpu().numpy()
            tags_pred=''
            for m in range(data["len_tokens"][0]):
                
                # print(dataset.idx2label(index_pred[0][m]))
                tags_pred += str(dataset.idx2label(index_pred[0][m]))
                if(m < data["len_tokens"][0]-1):
                    tags_pred += ' '
            # print(tags_pred)
            tags.append(tags_pred)
            id.append(data["id"][0])

        tags = np.array(tags)
        id = np.array(id)
        pred_id_tags = np.transpose(np.vstack((id, tags)))

    # TODO: write prediction to file (args.pred_file)
    pd.DataFrame(pred_id_tags).to_csv(args.pred_file, sep = ',',header=None, index=None)


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
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        required=True
    )
    parser.add_argument("--pred_file", type=Path, default="pred_slot.csv")

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)