from transformers import EsmTokenizer, EsmModel
import torch
from torch.nn.parallel import DataParallel
from torch import nn
import os
import random
import numpy as np

class ESMmodel_CNN(nn.Module):
    def __init__(self, model_path, hidden_size=640):
        super(ESMmodel_CNN, self).__init__()
        self.esmmodel=EsmModel.from_pretrained(model_path)
        self.tokenizer=EsmTokenizer.from_pretrained(model_path)
        self.hidden_size=hidden_size
        #self.CNN=nn.Conv1d(in_channels=self.hidden_size, out_channels=self.hidden_size, kernel_size=4, stride=1, padding=2)
        self.tanh=nn.Tanh()
        self.lstm=nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=1, batch_first=True)
        self.go_linear=nn.Linear(self.hidden_size, self.hidden_size)
        self.sample_div_linear=nn.Linear(self.hidden_size,self.hidden_size//2)
        self.go_div_linear=nn.Linear(self.hidden_size,self.hidden_size//2)
        self.linear_1=nn.Linear(self.hidden_size, self.hidden_size//2)
        self.relu=nn.ReLU()
        #self.linear_final=nn.Linear(self.hidden_size//2, 1)

        self.linear=nn.Linear(self.hidden_size, self.hidden_size)
        self.bin_linear=nn.Linear(self.hidden_size, 1)
        self.sig=nn.Sigmoid()

    def forward(self,sample_ids=None,sample_ids_att_mask=None,go_embeds=None,sample_go_ids=None,sample_go_ids_attn=None,type=1):
        if type==0:
            go_embed=self.esmmodel(input_ids=sample_go_ids,attention_mask=sample_go_ids_attn).pooler_output
            return go_embed
        sample_embeddings=self.esmmodel(input_ids=sample_ids,attention_mask=sample_ids_att_mask).pooler_output
        sample_embeddings=sample_embeddings.unsqueeze(0)
        c_0=torch.zeros([1,go_embeds.size()[0],self.hidden_size]).to(sample_embeddings.device)
        _,(pred_embed,c_n)=self.lstm(go_embeds,(sample_embeddings,c_0))
        #sample_embed_div=self.sample_div_linear(sample_embeddings)
        #go_embed_div=self.go_div_linear(go_embeds)
        #sample_embed_div=self.relu(sample_embed_div)
        #go_embed_div=self.relu(go_embed_div)
        #bin_embed=torch.cat([sample_embed_div,go_embed_div],dim=1)
        pred_embed=pred_embed.squeeze(0)
        pred_embed=self.go_linear(pred_embed)
        pred_embed=self.relu(pred_embed)
        pred_embed=self.bin_linear(pred_embed)
        #bin_embed=self.linear(bin_embed)
        #bin_embed=self.tanh(bin_embed)

        return self.sig(pred_embed)
'''class lstm_model(nn.Module):

    def __init__(self,hidden_size=640):
        super(lstm_model, self).__init__()
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.bin_linear = nn.Linear(hidden_size, 2)
        self.hidden_size=hidden_size
    def forward(self,sample_embeds,batch_go_embeds):
        sample_embeds=sample_embeds.unsqueeze(0)
        c_n=torch.zeros([1,sample_embeds.size()[1],self.hidden_size]).to(sample_embeds.device)
        output,(h_n,c_n)=self.lstm(batch_go_embeds,(sample_embeds,c_n))
        h_n=h_n.squeeze(0)
        binary_embed=self.linear(h_n)
        return self.bin_linear(binary_embed)'''