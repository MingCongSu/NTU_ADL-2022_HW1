from random import sample
from typing import List, Dict
import torch
from torch.utils.data import Dataset

from utils import Vocab
import utils


class SeqClsDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
    ):
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    # @property means "read-only"
    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn(self, samples: List[Dict]) -> Dict:
        #TODO: implement collate_fn
        text_list, intent_list, id_list= [], [], []

        if(len(samples[0]) == 2):
            for batch in samples:
                id_list.append(batch["id"])
                t = batch["text"].split()
                text_list.append(t)
            text_list = torch.tensor(self.vocab.encode_batch(text_list,self.max_len))
            batch = {"text": text_list, "id": id_list}
        else:            
            for batch in samples:
                i = self.label_mapping[batch["intent"]]
                intent_list.append(i)
                id_list.append(batch["id"])
                t = batch["text"].split()
                text_list.append(t)
            text_list = torch.tensor(self.vocab.encode_batch(text_list,self.max_len))
            intent_list = torch.tensor(intent_list)
            batch = {"text": text_list, "intent": intent_list, "id": id_list}
        
        return batch
    
    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]


class SeqTaggingClsDataset(SeqClsDataset):
    ignore_idx = -100

    def collate_fn(self, samples):
        # TODO: implement collate_fn
        tokens_list, tags_list, id_list, len_tokens, label = [], [], [], [], []

        if(len(samples[0]) == 2):
            for batch in samples:
                token = batch["tokens"]
                tokens_list.append(token)
                len_tokens.append(len(token))
                id_list.append(batch["id"])
            tokens_list = torch.tensor(self.vocab.encode_batch(tokens_list,self.max_len))
            # print(tokens_list)
            # print(id_list)
            batch = {"tokens": tokens_list, "id": id_list, "len_tokens": len_tokens}
        else:            
            for batch in samples:
                tag = []
                for m in range(len(batch["tags"])):
                    tag.append(self.label_mapping[batch["tags"][m]])
                tags_list.append(tag)
                
                token = batch["tokens"]
                tokens_list.append(token)

                id_list.append(batch["id"])

                len_tokens.append(len(token))

                label.append(batch["tags"])
            tokens_list = torch.tensor(self.vocab.encode_batch(tokens_list,self.max_len))
            tags_list = torch.tensor(utils.pad_to_len(tags_list, self.max_len, 1))
            batch = {"tokens": tokens_list, "tags": tags_list, "id": id_list, "len_tokens": len_tokens, "label": label}
        return batch
