## models.py

import torch
import torch.nn as nn

def get_activation(name: str):
    name = name.lower()
    if name == "relu": return nn.ReLU()
    if name == "tanh": return nn.Tanh()
    if name == "sigmoid": return nn.Sigmoid()
    raise ValueError(f"Unknown activation: {name}")

class BaseClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim=100, hidden_dim=64, num_layers=2,
                 dropout=0.5, activation="relu", bidirectional=False, rnn_type="rnn"):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.bidirectional = bidirectional
        self.rnn_type = rnn_type

        if rnn_type == "rnn":
            ## PyTorch vanilla RNN uses tanh inside
            self.rnn = nn.RNN(
                input_size=emb_dim, hidden_size=hidden_dim, num_layers=num_layers,
                nonlinearity="tanh", batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0, bidirectional=False
            )
            feat_dim = hidden_dim
        elif rnn_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=emb_dim, hidden_size=hidden_dim, num_layers=num_layers,
                batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=bidirectional
            )
            feat_dim = hidden_dim * (2 if bidirectional else 1)
        else:
            raise ValueError("rnn_type must be 'rnn' or 'lstm'")

        ## Two-layer MLP head with a configurable activation, then a single logit
        self.head = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        emb = self.embedding(x)
        emb = self.dropout(emb)

        if self.rnn_type == "rnn":
            out, h = self.rnn(emb)     ## h: (num_layers, B, H)
            feats = h[-1]              ## last layer hidden
        else:
            out, (h, c) = self.rnn(emb)
            if self.bidirectional:
                feats = torch.cat([h[-2], h[-1]], dim=-1)  ## concat fwd/bwd last layer
            else:
                feats = h[-1]

        logits = self.head(feats).squeeze(-1)
        return logits

def make_model(arch: str, vocab_size: int, activation: str, dropout=0.5):
    arch = arch.lower()
    if arch == "rnn":
        return BaseClassifier(vocab_size, activation=activation, dropout=dropout, rnn_type="rnn")
    if arch == "lstm":
        return BaseClassifier(vocab_size, activation=activation, dropout=dropout, rnn_type="lstm", bidirectional=False)
    if arch == "bilstm":
        return BaseClassifier(vocab_size, activation=activation, dropout=dropout, rnn_type="lstm", bidirectional=True)
    raise ValueError(f"Unknown architecture: {arch}")
