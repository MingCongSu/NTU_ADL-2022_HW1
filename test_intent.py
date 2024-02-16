import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
import numpy as np
import pandas as pd 
from dataset import SeqClsDataset
from model import SeqClassifier
from utils import Vocab
from torch.utils.data import DataLoader
from tqdm import tqdm


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data = json.loads(args.test_file.read_text())
    #data = {"test": json.loads(args.test_file.read_text())}
    dataset = SeqClsDataset(data, vocab, intent2idx, args.max_len)
    # TODO: crecate DataLoader for test dataset
    dataloader = DataLoader(dataset= dataset, collate_fn=dataset.collate_fn)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    device = args.device
    model = SeqClassifier(
        embeddings,
        args.hidden_size,
        args.num_layers,
        args.dropout,
        args.bidirectional,
        dataset.num_classes,
    )
    ckpt = torch.load(args.ckpt_path)
    # load weights into model
    model.load_state_dict(ckpt, strict=False)
    model.to(device)
    # TODO: predict dataset
    model.eval()
    with torch.no_grad():
        count = 0.0
        intent = ['intent']
        id = ['id']
        data_test=tqdm(dataloader)
        for _, data in enumerate(data_test):
            count+=1
            input=data["text"].to(device)
            pred=model(input)
            pred=pred.argmax(dim=1).item()
            intent_pred = dataset.idx2label(pred)
            intent.append(intent_pred)
            id.append(data["id"][0])
        pred_id_intent = [id, intent]
        pred_id_intent = np.transpose(np.array(pred_id_intent))
        #print(pred_id_intent)
    # TODO: write prediction to file (args.pred_file)
    pd.DataFrame(pred_id_intent).to_csv(args.pred_file, header=None, index=None)

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        required=True
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        required=True
    )
    parser.add_argument("--pred_file", type=Path, default="pred_intent.csv")

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
