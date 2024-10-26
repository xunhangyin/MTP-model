from transformers import EsmTokenizer, EsmModel
import torch
from torch.nn.parallel import DataParallel
from torch import nn
import os
import random
import chromadb
import math
import datetime
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Esmpredictor(object):
    def __init__(self, model_path,load_step,max_shot):
        self.path=model_path
        self.model = EsmModel.from_pretrained(model_path+str(load_step)+"/").to(device)
        self.tokenizer = EsmTokenizer.from_pretrained("./ESM/")
        self.max_shot = max_shot
        self.load_step=load_step


    def process_seq(self, batch_seqs, model):
        with torch.no_grad():
            batch_ids = self.tokenizer(batch_seqs, return_tensors="pt", padding=True, truncation=True)
            batch_ids.to(device)
            pooled_outputs = model(input_ids=batch_ids["input_ids"], attention_mask=batch_ids["attention_mask"],
                                   return_dict=True)
        return pooled_outputs
    def get_avg(self,id_seq_train, id_seq_test, anno_id_train, id_anno_test, go_ancestor, go_ancestor_test,collection):

        model = self.model
        model.eval()
        train_embed = {}
        num = 0
        for id_annos in anno_id_train:  ###{GO_id:[protein_id]}
            time_a = datetime.datetime.now()
            if num % 50 == 0:
                print(str(num) + "/" + str(len(anno_id_train)))
            seqs = []
            ids = []
            if len(anno_id_train[id_annos]) > self.max_shot:
                ids = random.sample(anno_id_train[id_annos], self.max_shot)
            else:
                ids = anno_id_train[id_annos]
            for id in ids:
                seqs.append(id_seq_train[id])
            time_b=datetime.datetime.now()

            anno_embed = torch.mean(self.process_seq(seqs, model).pooler_output, dim=0).tolist()
            #train_embed[id_annos] = anno_embed  ####显存问题,{GO_id:embed}
            time_c=datetime.datetime.now()

            collection.add(embeddings=anno_embed,documents=[id_annos],ids=[id_annos+"BPO"])
            time_d=datetime.datetime.now()

            num += 1
        test_embed = {}
        num=0
        for id in id_seq_test:
            num+=1
            if num%50==0:
                print(str(num)+"/"+str(len(id_seq_test)))
            seq = id_seq_test[id]
            test_embed[id] = self.process_seq([seq], model).pooler_output  ##{protein_id:embed}
        #torch.save(train_embed,self.path+str(self.load_step)+"/train_avg/train_embed.pth")
        torch.save(test_embed,self.path+str(self.load_step)+"/test_embed.pth")



    def predict(self, id_seq_train, id_seq_test, anno_id_train, id_anno_test, go_ancestor, go_ancestor_test):
        chroma_client = chromadb.PersistentClient(path=self.path+str(self.load_step)+"/chromadb/")
        collection = chroma_client.get_or_create_collection(name="emb_collection")
        files=os.listdir(self.path+str(self.load_step))
        if "test_embed.pth" not in files:
            self.get_avg(id_seq_train, id_seq_test, anno_id_train, id_anno_test, go_ancestor, go_ancestor_test,collection)
            pass
        #train_embed=torch.load(self.path+str(self.load_step)+"/train_avg/train_embed.pth")
        test_embed = torch.load(self.path+str(self.load_step)+"/test_embed.pth")

        res=collection.query(query_embeddings=test_embed[test_key].tolist(),n_results=128)
        GO_ids=res["documents"][0]
        distances=res["distances"][0]
        return GO_ids








