from transformers import EsmTokenizer, EsmModel
import torch
from torch.nn.parallel import DataParallel
from torch import nn
import os
import random
import numpy as np
from ESM_model import ESMmodel_CNN

# from ESM_mode import ESM_LSTM_model,ESM_MLP_model
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,4,5"
# device_lstm = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda")

class EsmTrainer(object):
    def __init__(self, model_path, output_dir, learning_rate, batch_size, num_epochs, weight_decay, accumulate_steps,
                 k_shot=8, eval_step=500, anchor_sample=1, max_n_ways=3, r=0.5,hidden_size=1280):
        self.model = ESMmodel_CNN(model_path).to(device)
        # self.LSTM_model=ESM_LSTM_model(model_path,1280,1280).to(torch.device("cuda:1"))
        # self.MLP_model=ESM_MLP_model(model_path,1280,1280).to(torch.device("cuda:6"))
        self.output_dir = output_dir
        self.tokenizer = EsmTokenizer.from_pretrained(model_path)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.weight_decay = weight_decay
        self.accumulate_steps = accumulate_steps
        self.eval_step = eval_step
        self.anchor_sample = anchor_sample
        self.k_shot = k_shot
        self.max_n_ways = max_n_ways
        self.r = r




    def get_batch(self, leaf_keys, id_anno, anno_id, step, go_ancestor):
        start_sample = step * self.anchor_sample
        num = -1
        batch_samples = []
        batch_go_samples = []
        leaf_key = ""
        for key in leaf_keys:
            num += 1
            if num < start_sample:
                continue
            if num >= start_sample and num < start_sample + self.anchor_sample:
                leaf_key = key
            if num > start_sample + self.anchor_sample:
                break
        
        labels = []
        go_sel = [leaf_key]
        num=0
        leaf_ancestors = go_ancestor[leaf_key]

        leaf_ancestors_sel = []
        for anc in leaf_ancestors:
            if anc in anno_id:
                leaf_ancestors_sel.append(anc)

        while len(batch_samples) < self.batch_size:
            num+=1
            if len(leaf_ancestors_sel) ==0 :
                if len(batch_samples)<self.batch_size /4 and num<30:
                    batch_samples.append(random.sample(anno_id[leaf_key], 1)[0])
                    labels.append(1)
                    continue
                sample_id = random.sample(list(id_anno.keys()), 1)[0]
                sample_go_ids = id_anno[sample_id]

                pub_ratio = 0
                for sample_go_id in sample_go_ids:
                    pub_ratio = max(self.get_pub_anc(leaf_key, sample_go_id, go_ancestor), pub_ratio)

                batch_samples.append(sample_id)

                labels.append(pub_ratio)


            else:
                if len(batch_samples) < self.batch_size / 4 and num<20:
    
                    batch_samples.append(random.sample(anno_id[leaf_key], 1)[0])
                    
                    labels.append(1)
                    continue
                '''if len(batch_samples)<self.batch_size /4 and num<50:
                    #batch_samples.append(random.sample(anno_id[random.sample(leaf_ancestors_sel, 1)[0]], 1)[0])
                    anc=random.sample(leaf_ancestors_sel, 1)[0]
                    batch_samples.append(random.sample(anno_id[anc], 1)[0])
                    labels.append(1)
                    continue'''
                    
                sample_id = random.sample(list(id_anno.keys()), 1)[0]
                sample_go_ids = id_anno[sample_id]
    
                
                pub_ratio = 0
                for sample_go_id in sample_go_ids:
                    pub_ratio = max(self.get_pub_anc(leaf_key, sample_go_id, go_ancestor), pub_ratio)
                batch_samples.append(sample_id)
                
                labels.append(pub_ratio)

        for i in range(self.batch_size):
            ids = anno_id[leaf_key]
            if len(ids) < self.k_shot:
                batch_go_samples.append(ids)
            else:
                batch_go_samples.append(random.sample(ids, self.k_shot))
        
        return batch_samples, batch_go_samples,labels

    def process_samples(self, batch_samples, batch_go_samples,id_seq,model):
        batch_sample_seqs=[]

        for sample in batch_samples:
            batch_sample_seqs.append(id_seq[sample])
        num=0
        for go_samples in batch_go_samples:
            batch_go_seqs=[]
            for go_sample in go_samples:
                batch_go_seqs.append(id_seq[go_sample])
            batch_go_ids=self.tokenizer(batch_go_seqs, return_tensors="pt", padding=True, truncation=True)
            input_ids=batch_go_ids["input_ids"].to(device)
            attention_mask=batch_go_ids["attention_mask"].to(device)
            go_embeds=model(sample_go_ids=input_ids,sample_go_ids_attn=attention_mask,type=0)
            #go_embeds=torch.mean(go_embeds,dim=0)
            if num==0:
                batch_go_embeds=go_embeds.unsqueeze(0)

            else:
                batch_go_embeds=torch.cat([batch_go_embeds,go_embeds.unsqueeze(0)],dim=0)
            num+=1

        batch_sample_ids=self.tokenizer(batch_sample_seqs, return_tensors="pt", padding=True, truncation=True)
        sample_input_ids=batch_sample_ids["input_ids"].to(device)
        sample_attention_mask=batch_sample_ids["attention_mask"].to(device)
        bin_embed=model(sample_ids=sample_input_ids,sample_ids_att_mask=sample_attention_mask,go_embeds=batch_go_embeds)

        return bin_embed


    # def get_support_ancestor(self,support_set)
    #    for key in support_set:
    #def get_logit(self,sample_embeds,batch_go_embeds):
        

    def get_pub_anc(self, key_support, key_query, go_ancestor):
        num = 0
        ancestor_support = go_ancestor[key_support]
        ancestor_query = go_ancestor[key_query]
        for ancestor in ancestor_support:
            if ancestor in ancestor_query:
                num += 1
        if len(ancestor_support) == 0:
            return 0

        return num / len(ancestor_support)

    def compute_loss(self, support_embeds, query_embeds, go_ancestor):

        loss_total = 0

        for key_query in query_embeds:

            labels = []
            probs = []
            pub_ratios = []
            loss = 0
            query_embed = query_embeds[key_query]
            for key_support in support_embeds:
                if key_query == key_support:
                    labels.append(1)
                    pub_ratios.append(1.0)
                    probs.append(self.get_logit(support_embeds[key_support], query_embed))
                else:
                    pub_ratio = self.get_pub_anc(key_support, key_query, go_ancestor)
                    labels.append(0)
                    pub_ratios.append(pub_ratio)
                    probs.append(self.get_logit(query_embed, support_embeds[key_support]))
            prob_total = 0
            for prob, pub_ratio, label in zip(probs, pub_ratios, labels):
                if label == 1:
                    prob_total += prob
                    continue
                prob_total = prob_total + prob * (1 - pub_ratio)
            for i in range(len(labels)):
                if labels[i] == 1:
                    loss_total += -torch.log(probs[i] / prob_total)

        return loss_total / len(query_embeds)

    def train(self, id_seq, id_anno, anno_id, id_seq_test, id_anno_test, anno_id_test, go_ancestor, go_ancestor_test,
              leaf_keys, use_large=False):
        if use_large:
            peft_config = LoraConfig(
                r=4, lora_alpha=32, lora_dropout=0.02, inference_mode=False,
                target_modules=["query", "key", "value", "dense"]
            )
            model = get_peft_model(self.model, peft_config)

            model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
            model.train()

        else:
            model = DataParallel(self.model, device_ids=[ 0, 1,2,3])
            '''for n,p in model_lstm.named_parameters():
                if "esm" in n:
                    if "pooler" not in n:
                        p.requires_grad=False
            switch=0
            for n,p in model_MLP.named_parameters():
                if "27" in n:
                    switch=1
                if switch==0:
                    p.requires_grad=False'''
        #leaf_keys=leaf_keys[95:]
        switch=1
        for n,p in model.named_parameters():
            if "8" in n:
                switch=0
            if switch==1:
                p.requires_grad=False
        
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        total_step = 0

        for epoch in range(self.num_epochs):
            loss_avg = 0
            for step in range(len(leaf_keys) // self.anchor_sample):

                model.train()

                batch_samples,batch_go_samples,labels= self.get_batch(leaf_keys, id_anno, anno_id, step,go_ancestor)
                logits=self.process_samples(batch_samples, batch_go_samples, id_seq, model)
                #logits=self.lstm(sample_embeds,batch_go_embeds)
                logits=logits.squeeze(1)
                # support_set = self.get_support_set(anno_id, batch_annos, batch_keys)
                labels=torch.tensor(labels,dtype=torch.float32).to(device)

                mse_loss = nn.L1Loss(reduction="mean")
                loss=mse_loss(logits,labels)
                if np.isnan(loss.item()):
                    continue

                loss.backward()
                
                optimizer.step()
                optimizer.zero_grad()
                loss_avg += loss.item()

                if total_step % 50 == 0:
                    loss_avg = loss_avg / 50
                    print("epoch:" + str(epoch) + " step:" + str(step) + "/" + str(
                        len(leaf_keys) // self.anchor_sample) + " loss:" + str(loss_avg))
                    loss_avg = 0
                if total_step % self.eval_step == 0 and step != 0:
                    model.eval()
                    #model.module.save_pretrained(self.output_dir + str(total_step))
                    torch.save(model.module,self.output_dir + str(total_step))
                    #self.tokenizer.save_pretrained(self.output_dir + str(total_step))
                    print("model saved")
                total_step += 1








