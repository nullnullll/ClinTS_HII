import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from Baseline.transformer import MultiHeadedAttention, PositionwiseFeedForward, Encoder, EncoderLayer
import copy

class enc_mtan_GRU(nn.Module):
    def __init__(self, input_dim, query,
                 embed_time=16, num_heads=1, nhidden=256, device='cuda'):
        super(enc_mtan_GRU, self).__init__()
        self.embed_time = embed_time
        self.dim = input_dim
        self.device = device
        self.nhidden = nhidden
        self.query = query

        self.ENCmTAN = multiTimeAttention(input_dim, embed_time, num_heads)
        self.encoder = nn.GRU(input_dim * num_heads, nhidden, bidirectional=True, batch_first=True)

        self.periodic = nn.Linear(1, embed_time - 1)
        self.linear = nn.Linear(1, 1)


    def learn_time_embedding(self, tt):
        tt = tt.to(self.device)
        tt = tt.unsqueeze(-1)
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)

    def forward(self, x, time_steps, causal_mask=None):
        time_steps = time_steps.cpu()
        mask = x[:, :, self.dim:]

        key = self.learn_time_embedding(time_steps).to(self.device)
        query = self.learn_time_embedding(self.query.unsqueeze(0)).to(self.device)

        ENCinput = self.ENCmTAN(query, key, x[:, :, :self.dim], mask, causal_mask)

        ENCoutput, _ = self.encoder(ENCinput)

        return ENCoutput

class enc_mtan_GRU_treatment_emb(nn.Module):
    def __init__(self, input_dim, query,
                 embed_time=16, num_heads=1, nhidden=256, device='cuda'):
        super(enc_mtan_GRU_treatment_emb, self).__init__()
        self.embed_time = embed_time
        self.dim = input_dim
        self.device = device
        self.nhidden = nhidden
        self.query = query

        self.ENCmTAN = multiTimeAttention(input_dim, embed_time, num_heads)
        self.encoder = nn.GRU( input_dim * num_heads, nhidden, bidirectional=True, batch_first=True)

        self.periodic = nn.Linear(1, embed_time - 1)
        self.linear = nn.Linear(1, 1)

        # event type embedding
        self.event_emb = nn.Linear(13, embed_time, bias=False)

        self.event_embedding_to_effect = nn.Linear(embed_time, 13)

        c = copy.deepcopy
        EncInputDim = embed_time
        attn = MultiHeadedAttention(4, EncInputDim)
        ff = PositionwiseFeedForward(EncInputDim, d_ff=100, dropout=0.1)
        self.ENCevent = Encoder(EncoderLayer(EncInputDim, c(attn), c(ff), 0.1), 4)


    def learn_time_embedding(self, tt):
        tt = tt.to(self.device)
        tt = tt.unsqueeze(-1)
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)

    def forward(self, x, time_steps, causal_mask=None):
        time_steps = time_steps.cpu()
        mask = x[:, :, self.dim:]

        key = self.learn_time_embedding(time_steps).to(self.device)
        query = self.learn_time_embedding(self.query.unsqueeze(0)).to(self.device)

        event_embedding = self.event_emb(x[:, :, 12:self.dim]) + key
        event_value = self.ENCevent(event_embedding)
        effect_representation = self.event_embedding_to_effect(event_value)

        physiological_status = self.ENCmTAN(query, key, x[:, :, :12], mask[:, :, :12], causal_mask=None)
        treatment_effect = self.ENCmTAN(query, key, effect_representation, mask[:, :, 12:], causal_mask=causal_mask)

        ENCoutput, _ = self.encoder(torch.cat([physiological_status, treatment_effect], -1))

        return ENCoutput

class enc_mtan_GRU_withoutirr(nn.Module):
    def __init__(self, input_dim, query,
                 embed_time=16, num_heads=1, nhidden=256, device='cuda'):
        super(enc_mtan_GRU_withoutirr, self).__init__()
        self.embed_time = embed_time
        self.dim = input_dim
        self.device = device
        self.nhidden = nhidden
        self.query = query

        self.ENCmTAN = multiTimeAttention(input_dim, embed_time, num_heads)
        self.encoder = nn.GRU(embed_time, nhidden, bidirectional=True, batch_first=True)

        self.embedding = nn.Linear(23, embed_time, bias=False)

        self.periodic = nn.Linear(1, embed_time - 1)
        self.linear = nn.Linear(1, 1)


    def learn_time_embedding(self, tt):
        tt = tt.to(self.device)
        tt = tt.unsqueeze(-1)
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)

    def forward(self, x, time_steps, causal_mask=None):

        embedding = self.embedding(x[:, :, :self.dim])

        ENCoutput, _ = self.encoder(embedding)

        return ENCoutput

class dec_mtan(nn.Module):
    def __init__(self, input_dim, ENCoutput_dim , query,
                 embed_time=16, num_heads=1,  device='cuda'):
        super(dec_mtan, self).__init__()
        self.embed_time = embed_time
        self.dim = input_dim
        self.device = device
        self.query = query

        self.DECmTAN = multiTimeAttention(ENCoutput_dim, embed_time, num_heads)

        self.periodic = nn.Linear(1, embed_time - 1)
        self.linear = nn.Linear(1, 1)

        self.OutputLayer = nn.Linear(ENCoutput_dim, input_dim)

    def learn_time_embedding(self, tt):
        tt = tt.to(self.device)
        tt = tt.unsqueeze(-1)
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)

    def forward(self, ENCoutput, time_steps, causal_mask=None):
        time_steps = time_steps.cpu()

        key = self.learn_time_embedding(time_steps).to(self.device)
        query = self.learn_time_embedding(self.query.unsqueeze(0)).to(self.device)

        out = self.DECmTAN(key, query, ENCoutput)

        return self.OutputLayer(out)

class dec_mtan_witouirr(nn.Module):
    def __init__(self, input_dim, ENCoutput_dim , query,
                 embed_time=16, num_heads=1,  device='cuda'):
        super(dec_mtan_witouirr, self).__init__()
        self.embed_time = embed_time
        self.dim = input_dim
        self.device = device
        self.query = query

        self.DECmTAN = multiTimeAttention(ENCoutput_dim, embed_time, num_heads)

        self.periodic = nn.Linear(1, embed_time - 1)
        self.linear = nn.Linear(1, 1)

        self.OutputLayer = nn.Linear(ENCoutput_dim, input_dim)

    def learn_time_embedding(self, tt):
        tt = tt.to(self.device)
        tt = tt.unsqueeze(-1)
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)

    def forward(self, ENCoutput, time_steps, causal_mask=None):

        return self.OutputLayer(ENCoutput)

def make_Encoder_GRU(args, d_model):
    if args.with_treatment:
        if args.withoutirr:
            model = enc_mtan_GRU_withoutirr(
                d_model, torch.linspace(0, 1., 128),
                args.embed_time, args.num_heads, args.rec_hidden)
        else:
            if not args.withoutheter:
                model = enc_mtan_GRU_treatment_emb(
                    d_model, torch.linspace(0, 1., 128),
                    args.embed_time, args.num_heads, args.rec_hidden)
            else:
                model = enc_mtan_GRU(
                    d_model, torch.linspace(0, 1., 128),
                    args.embed_time, args.num_heads, args.rec_hidden)

    else:
        model = enc_mtan_GRU(
            d_model, torch.linspace(0, 1., 128),
            args.embed_time, args.num_heads, args.rec_hidden)
    return model

def make_Decoder_GRU(args, d_model):
    if args.withoutirr:
        model = dec_mtan_witouirr(
            d_model, args.rec_hidden * 2, torch.linspace(0, 1., 128),
            args.embed_time, 1)
    else:
        model = dec_mtan(
            d_model, args.rec_hidden * 2, torch.linspace(0, 1., 128),
            args.embed_time, 1)
    return model

class Classifier(nn.Module):

    def __init__(self, hidden_dim, length, cls_dim =300, N=2, args=None ):
        super(Classifier, self).__init__()

        self.pool = nn.MaxPool1d(length)

        self.classifier = nn.Linear(hidden_dim, N)

    def forward(self, ENCoutput, args):

        out = self.pool(ENCoutput.transpose(1,2))

        return self.classifier(out.squeeze(-1))

class multiTimeAttention(nn.Module):

    def __init__(self, input_dim,
                 embed_time=16, num_heads=1):
        super(multiTimeAttention, self).__init__()
        assert embed_time % num_heads == 0
        self.embed_time = embed_time
        self.embed_time_k = embed_time // num_heads
        self.h = num_heads
        self.dim = input_dim

        self.linears = nn.ModuleList([nn.Linear(embed_time, embed_time),
                                      nn.Linear(embed_time, embed_time)])

    def attention(self, query, key, value, mask=None, causal_mask=None, dropout=None):
        dim = value.size(-1)
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(d_k)
        scores = scores.unsqueeze(-1).repeat_interleave(dim, dim=-1)
        if causal_mask is not None:
            scores = scores.masked_fill(causal_mask.unsqueeze(-4) == 0, -1e9)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(-3) == 0, -1e9)
        p_attn = F.softmax(scores, dim=-2)
        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.sum(p_attn * value.unsqueeze(-3), -2), p_attn

    def forward(self, query, key, value, mask=None, causal_mask=None, dropout=None):
        batch, seq_len, dim = value.size()
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        value = value.unsqueeze(1)
        query, key = [l(x).view(x.size(0), -1, self.h, self.embed_time_k).transpose(1, 2)
                      for l, x in zip(self.linears, (query, key))]
        x, _ = self.attention(query, key, value, mask, causal_mask, dropout)
        x = x.transpose(1, 2).contiguous() \
            .view(batch, -1, self.h * dim)
        return x