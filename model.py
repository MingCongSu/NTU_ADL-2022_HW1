from typing import Dict
from numpy import transpose

import torch
import torch.nn as nn
from torch.nn import Embedding


class SeqClassifier(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int, 
    ) -> None:
        super(SeqClassifier, self).__init__()
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        # TODO: model architecture
        self.embed_dim = embeddings.size(1)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.dr=nn.Dropout(dropout)
        self.relu=nn.ReLU()
        self.batch_norm1 = torch.nn.BatchNorm1d(1024)
        self.batch_norm2 = torch.nn.BatchNorm1d(128)
        self.batch_norm3 = torch.nn.BatchNorm1d(128)
        self.li1=nn.Linear(hidden_size*2,512)
        self.li2=nn.Linear(512,256)
        self.li3=nn.Linear(256,128)        
        self.lstm = nn.LSTM(self.embed_dim, hidden_size=self.hidden_size, num_layers=self.num_layers, dropout=self.dropout, bidirectional=bidirectional)
        self.gru = nn.GRU(self.embed_dim, hidden_size=self.hidden_size, num_layers=self.num_layers, dropout=(self.dropout if num_layers > 1 else 0), bidirectional=bidirectional)
        self.li4= nn.Linear(128, num_class)
        self.li5 = nn.Linear(512, num_class)
        self.layer_norm1 = torch.nn.LayerNorm(embeddings.shape[-1])
        self.layer_norm2 = torch.nn.LayerNorm(1024)
        self.layer_norm3 = torch.nn.LayerNorm(512)
    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        raise NotImplementedError

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        embedding = self.embed(batch)
        out,_= self.lstm(embedding.transpose(0,1))
        out=out.transpose(0,1)[:,1,:]
        out = self.batch_norm1(out)
        out = self.layer_norm2(out)
        out=self.li1(out)
        out=self.dr(out)
        out=self.relu(out)
        out=self.layer_norm3(out)
        out=self.li2(out)
        out=self.dr(out)
        out=self.relu(out)
        out=self.li3(out)
        out=self.relu(out)
        out=self.batch_norm2(out)
        out=self.li4(out)
        return out


class SeqTagger(SeqClassifier):
    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        x = self.embed(batch)
        x = self.layer_norm1(x)
        x,_=self.gru(x.transpose(0,1))
        x = (x.transpose(0,1).transpose(1, 2))
        x = self.batch_norm1(x)
        x = x.transpose(1,2)
        x = self.layer_norm2(x)
        x = self.li1(x)
        x = self.relu(x)
        x = self.li5(x)
        return x
