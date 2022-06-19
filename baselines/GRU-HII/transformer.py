import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class Encoder(nn.Module):

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask=None):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderLayer(nn.Module):

    def __init__(self, size ,self_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)

    def forward(self, x, mask=None):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class enc_mtan_pretrain(nn.Module):

    def __init__(self, encoder , input_dim, query,
                 embed_time=16, num_heads=1,  device='cuda'):
        super(enc_mtan_pretrain, self).__init__()
        self.embed_time = embed_time

        self.dim = input_dim
        self.device = device
        self.query = query

        self.ENCmTAN = multiTimeAttention( input_dim, embed_time, num_heads)
        self.encoder = encoder

        self.periodic = nn.Linear(1, embed_time - 1)
        self.linear = nn.Linear(1, 1)

    def learn_time_embedding(self, tt):
        tt = tt.to(self.device)
        tt = tt.unsqueeze(-1)
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)

    def forward(self, x, time_steps, causal_mask=None ):
        time_steps = time_steps.cpu()
        mask = x[:, :, self.dim:]


        key = self.learn_time_embedding(time_steps).to(self.device)
        query = self.learn_time_embedding(self.query.unsqueeze(0)).to(self.device)

        ENCinput = self.ENCmTAN(query, key, x[:, :, :self.dim], mask, causal_mask)
        ENCoutput = self.encode(ENCinput)

        return  ENCoutput

    def encode(self, ENCinput):
        return self.encoder(ENCinput)


class dec_mtan_pretrain(nn.Module):

    def __init__(self, decoder, input_dim, outputdim, query,
                 embed_time=16, num_heads=1, device='cuda'):
        super(dec_mtan_pretrain, self).__init__()
        self.embed_time = embed_time

        self.dim = input_dim
        self.device = device
        self.query = query

        self.DECmTAN = multiTimeAttention(input_dim, embed_time, num_heads)
        self.decoder = decoder
        self.output = nn.Sequential(
            nn.Linear(input_dim, 50),
            nn.ReLU(),
            nn.Linear(50, outputdim))

        self.periodic = nn.Linear(1, embed_time - 1)
        self.linear = nn.Linear(1, 1)

    def learn_time_embedding(self, tt):
        tt = tt.to(self.device)
        tt = tt.unsqueeze(-1)
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)

    def forward(self, ENCoutput, time_steps, mask=None, causal_mask=None):
        time_steps = time_steps.cpu()

        key = self.learn_time_embedding(time_steps).to(self.device)
        query = self.learn_time_embedding(self.query.unsqueeze(0)).to(self.device)

        DECinput = self.DECmTAN(key, query, ENCoutput,mask)
        DECoutput = self.decoder(DECinput)
        DECoutput = self.output(DECoutput)
        return DECoutput

    def decode(self, DECinput):
        return self.decoder(DECinput)



class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


def make_Encoder(config):
    Nenc=config["Nenc"]
    d_model=config["d_model"]
    d_ff=config["d_ff"]
    h=config["num_heads"]
    dropout=config["dropout_rate"]
    num_heads=config["num_mTAN_heads"]
    embed_time=config["embed_time"]
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model * num_heads)
    ff = PositionwiseFeedForward(d_model * num_heads, d_ff, dropout)
    model = enc_mtan_pretrain(
        Encoder(EncoderLayer(d_model * num_heads, c(attn), c(ff), dropout), Nenc),
        d_model, torch.linspace(0, 1., 128),
        embed_time, num_heads)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model

def make_Decoder(config):
    Ndec=config["Ndec"]
    d_model=config["d_model"]
    d_ff=config["d_ff"]
    h=config["num_heads"]
    dropout=config["dropout_rate"]
    num_heads=config["num_mTAN_heads"]
    embed_time=config["embed_time"]
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model * num_heads)
    ff = PositionwiseFeedForward(d_model * num_heads, d_ff, dropout)
    model = dec_mtan_pretrain(
        Decoder(DecoderLayer(d_model * num_heads, c(attn), c(ff), dropout), Ndec),
        d_model* num_heads, d_model,torch.linspace(0, 1., 128),
        embed_time, 1)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model

def evaluate_maskedmodel(enc, dec, test_loader, args=None, classifier=None,
                        dim=12, device='cuda', reconst=False, num_sample=1):
    pred = []
    true = []
    test_loss = 0
    for test_batch, label in test_loader:
        test_batch, label = test_batch.to(device), label.to(device)
        batch_len = test_batch.shape[0]
        observed_data, observed_mask, observed_tp= \
            test_batch[:, :, :dim], test_batch[:, :, dim:2 * dim], test_batch[:, :, -1]
        zeros = torch.zeros(observed_mask.shape).to(device)
        ones = torch.ones(observed_mask.shape).to(device)
        pretrain_mask = torch.rand(observed_mask.shape).to(device)
        pretrain_mask = torch.where(pretrain_mask > 0.5, ones, zeros).to(device)
        masked_position = ~pretrain_mask.bool()

        combined_mask = (pretrain_mask.bool() & observed_mask.bool()).float()
        masked_position = (masked_position & observed_mask.bool()).float()

        masked_val = observed_data.masked_fill(masked_position == 0, 0)
        observed_data = observed_data.masked_fill(combined_mask == 0, 0.)


        with torch.no_grad():
            ENCoutput = enc(torch.cat((observed_data, combined_mask), 2), observed_tp)
            DECoutput = dec(ENCoutput, observed_tp)

            out = DECoutput.masked_fill(masked_position == 0, 0)
            test_loss += nn.MSELoss()(out, masked_val).item() * batch_len * num_sample
        pred.append(out.cpu().numpy())
    pred = np.concatenate(pred, 0)
    pred = np.nan_to_num(pred)
    return test_loss/pred.shape[0]