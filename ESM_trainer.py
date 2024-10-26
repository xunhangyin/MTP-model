from transformers import EsmTokenizer, EsmModel
import torch
from torch.nn.parallel import DataParallel
from torch import nn
import os
import random
import numpy as np
from peft import get_peft_model, LoraConfig, TaskType
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class EsmTrainer(object):
    def __init__(self, model_path, output_dir, learning_rate, batch_size, num_epochs, weight_decay, accumulate_steps,
                 k_shot=8, eval_step=500, anchor_sample=1, max_n_ways=3, r=0.5):
        self.model = EsmModel.from_pretrained(model_path).to(device)
        #self.model=ESM_model(model_path,2560).to(device)
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

    def get_batch(self, id_seq, anno_id, step, sample_num):
        start_sample = step * self.anchor_sample
        num = 0
        batch_anno_id = {}
        for key in anno_id:
            if num < start_sample:
                continue
            if num >= start_sample and num <= start_sample + self.anchor_sample:
                if len(anno_id[key]) <= self.k_shot:
                    batch_anno_id[key] = anno_id[key]
                else:
                    batch_anno_id[key] = random.sample(anno_id[key], self.k_shot)
            if num > start_sample + self.anchor_sample:
                break
            num += 1

        while len(batch_anno_id) < self.batch_size:
            key = random.choice(list(anno_id.keys()))
            if key in batch_anno_id:
                continue
            if len(anno_id[key]) <= self.k_shot:
                batch_anno_id[key] = anno_id[key]
            else:
                batch_anno_id[key] = random.sample(anno_id[key], self.k_shot)
        return batch_anno_id

    def process_seq(self, batch_seqs, model):
        batch_ids = self.tokenizer(batch_seqs, return_tensors="pt", padding=True, truncation=True)
        pooled_outputs = model(input_ids=batch_ids["input_ids"], attention_mask=batch_ids["attention_mask"])

        return pooled_outputs

    def get_embed(self, batch_anno_id, id_seq, model):
        query_anno_embed = {}
        support_embed = {}
        for anno in batch_anno_id:
            anno_seqs = []
            for id in batch_anno_id[anno]:
                anno_seqs.append(id_seq[id])
            anno_embeds = self.process_seq(anno_seqs, model)
            query_index = random.randint(0, len(anno_seqs) - 1)
            query_embed = anno_embeds[query_index]
            query_anno_embed[anno] = query_embed
            anno_embeds = torch.cat((anno_embeds[0:query_index], anno_embeds[query_index + 1:]), dim=0)
            support_embed[anno] = torch.mean(anno_embeds, dim=0)
        return support_embed, query_anno_embed

    # def get_support_ancestor(self,support_set)
    #    for key in support_set:
    def get_logit(self, support_vec_embed, query_embed):
        pdist = nn.PairwiseDistance(p=2)
        dist = pdist(query_embed, support_vec_embed)
        return torch.exp(-dist)

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

        loss_total=0

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
            prob_total=0
            for prob, pub_ratio,label in zip(probs,pub_ratios,labels):
                if label==1:
                    prob_total+=prob
                    continue
                prob_total=prob_total+prob*(1-pub_ratio)
            for i in range(len(labels)):
                if labels[i]==1:
                    loss_total+=-torch.log(probs[i]/prob_total)

        return loss_total/len(query_embeds)

    def train(self, id_seq, id_anno, anno_id, id_seq_test, id_anno_test, anno_id_test, go_ancestor, go_ancestor_test,
              use_large=False):
        if use_large:
            peft_config = LoraConfig(
                r=4, lora_alpha=32, lora_dropout=0.02, inference_mode=False,
                target_modules=["query", "key", "value", "dense"]
            )
            model = get_peft_model(self.model, peft_config)

            model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
            model.train()

        else:
            model = DataParallel(self.model, device_ids=[0,1,2,3,4,5,6])

        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        total_step = 0

        for epoch in range(self.num_epochs):
            loss_avg = 0
            for step in range(len(anno_id) // self.anchor_sample):
                model.train()
                keys = list(anno_id.keys())
                batch_anno_id = self.get_batch(id_seq, anno_id, step, len(id_seq))
                # support_set = self.get_support_set(anno_id, batch_annos, batch_keys)
                support_embeds, query_embeds = self.get_embed(batch_anno_id, id_seq, model)
                loss = self.compute_loss(support_embeds, query_embeds, go_ancestor)
                if np.isnan(loss.item()):
                    continue
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                loss_avg += loss.item()
                if total_step % 100 == 0:
                    loss_avg = loss_avg / 50
                    print("epoch:" + str(epoch) + " step:" + str(step) + "/" + str(
                        len(anno_id) // self.anchor_sample) + " loss:" + str(loss_avg))
                    loss_avg = 0
                if total_step % self.eval_step == 0 and step != 0:
                    model.eval()
                    model.module.save_pretrained(self.output_dir + str(total_step) + "/")
                    self.tokenizer.save_pretrained(self.output_dir + str(total_step) + "/")
                    print("model saved")
                total_step += 1








