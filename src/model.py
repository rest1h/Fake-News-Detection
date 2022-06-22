import numpy as np
import pandas as pd
import partial
import torch
from torch import nn
from torch.nn import Linear
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_max_pool as gmp
from transformers import AutoModel, AutoTokenizer


def embed_tweets(
    tweet_df: pd.DataFrame, text_embedding_model_id: str = "xlm-roberta-base"
) -> pd.DataFrame:

    # Load text embedding model
    model_id = text_embedding_model_id
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)

    # Define embedding function
    embed = partial(embed_text, tokenizer=tokenizer, model=model)

    # Embed tweet text using the pretrained transformer
    text_embs = tweet_df.text.progress_apply(embed)
    tweet_df["text_emb"] = text_embs

    return tweet_df


def embed_text(text: str, tokenizer, model) -> np.ndarray:
    """
    Extract a text embedding.

    Args:
        text (str): The text to embed.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use.
        model (transformers.PreTrainedModel): The model to use.

    Returns:
        np.ndarray: The embedding of the text.
    """
    import torch

    with torch.no_grad():
        inputs = tokenizer(text, truncation=True, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        result = model(**inputs)
        return result.pooler_output[0].cpu().numpy()


class GNN(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super().__init__()

        # Graph Convolutions
        self.conv1 = GATConv(in_channels, hidden_channels)
        self.conv2 = GATConv(hidden_channels, hidden_channels)
        self.conv3 = GATConv(hidden_channels, hidden_channels)

        # Readout
        self.lin_news = Linear(in_channels, hidden_channels)
        self.lin0 = Linear(hidden_channels, hidden_channels)
        self.lin1 = Linear(2 * hidden_channels, out_channels)

        self._emb = None

    def forward(self, x: torch.tensor, edge_index: torch.tensor, batch: torch.tensor) -> torch.tensor:
        # Graph Convolutions
        h = self.conv1(x, edge_index).relu()
        h = self.conv2(h, edge_index).relu()
        h = self.conv3(h, edge_index).relu()

        # Pooling
        h = gmp(h, batch)

        # Readout
        h = self.lin0(h).relu()

        # According to UPFD paper: Include raw word2vec embeddings of news
        # This is done per graph in the batch
        root = (batch[1:] - batch[:-1]).nonzero(as_tuple=False).view(-1)
        root = torch.cat([root.new_zeros(1), root + 1], dim=0)
        # root is e.g. [   0,   14,   94,  171,  230,  302, ... ]
        news = x[root]
        news = self.lin_news(news).relu()

        out = self.lin1(torch.cat([h, news], dim=-1))
        self._emb = out.clone()
        # print(out)
        return torch.sigmoid(out)

    @property
    def emb(self) -> torch.tensor:
        return self._emb

    @emb.setter
    def emb(self, value) -> None:
        self._emb = value


class Attention(nn.Module):
    def __init__(self, enc_hid_dim: int, dec_hid_dim: int):
        super().__init__()

        self.attn = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, hidden: torch.tensor, encoder_outputs: torch.tensor) -> torch.tensor:
        # hidden = [batch size, dec hid dim]
        # encoder_outputs = [src len, batch size, enc hid dim * 2]
        src_len = encoder_outputs.shape[0]

        # repeat decoder hidden state src_len times
        # hidden = [batch size, src len, dec hid dim]
        # encoder_outputs = [batch size, src len, enc hid dim * 2]
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        encoder_outputs = encoder_outputs.unsqueeze(1).repeat(1, src_len, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # energy = [batch size, src len, dec hid dim]
        energy = self.tanh(self.attn(torch.cat([hidden, encoder_outputs], dim=2)))

        # attention= [batch size, src len]
        attention = self.v(energy).squeeze(2)
        return attention


class GNNWithAttentionFusion(GNN):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, device=None):
        super(GNNWithAttentionFusion, self).__init__(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels
        )

        self.attention_1 = Attention(hidden_channels, out_channels)
        self.attention_2 = Attention(hidden_channels, out_channels)
        self.root_size = None

        self.sigmoid = nn.Sigmoid()
        self._emb = None
        self.hidden_channels = hidden_channels
        self.device = device or "cpu"

    def forward(self, x: torch.tensor, edge_index: torch.tensor, batch: torch.tensor) -> torch.tensor:
        # Graph Convolutions
        h = self.conv1(x, edge_index).relu()
        h = self.conv2(h, edge_index).relu()
        h = self.conv3(h, edge_index).relu()

        # Pooling
        h = gmp(h, batch)

        # Readout
        h = self.lin0(h).relu()

        # According to UPFD paper: Include raw word2vec embeddings of news
        # This is done per graph in the batch
        root = (batch[1:] - batch[:-1]).nonzero(as_tuple=False).view(-1)
        root = torch.cat([root.new_zeros(1), root + 1], dim=0)
        if root.shape[0] < self.hidden_channels:
            self.root_size = root.shape[0]
            zero_pad = torch.zeros(self.hidden_channels - root.shape[0], dtype=torch.long).to(self.device)
            root = torch.cat([root, zero_pad])
            zero_pad = torch.zeros((zero_pad.shape[0], 128), dtype=torch.long).to(self.device)
            h = torch.cat([h, zero_pad], dim=0)

        # root is e.g. [   0,   14,   94,  171,  230,  302, ... ]
        news = x[root]
        news = self.lin_news(news).relu()

        fused_h = self.attention_1(h, news)
        fused_news = self.attention_2(news, h)
        out = torch.cat([fused_h, fused_news], dim=-1).view(-1, 256)
        out = self.lin1(out)
        self._emb = out.clone()

        if self.root_size:
            out = self.sigmoid(out[: self.root_size])
            self.root_size = None
            return out

        return self.sigmoid(out)
