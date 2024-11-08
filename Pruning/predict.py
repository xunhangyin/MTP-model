import torch
import argparse
from get_parse import parser_add_main_args
from GO_process import get_ancestor
import json
import random
from get_data_new import get_data
import torch
import math
train_fasta_path="./finetune_data/data_heal/PDB_train_sequences.fasta"
test_fasta_path="./finetune_data/data_heal/PDB_test_sequences.fasta"
id_seq,id_seq_test,id_anno,id_anno_test,anno_id,anno_id_test,go_ancestor,go_ancestor_test=get_data(train_fasta_path,test_fasta_path)
with open("./Can_anc_files/go_ancestor_BPO.json","r") as f:   ###edit the path to select BP,CC,MF
    go_ancestor_BPO=json.load(f)
with open("./Can_anc_files/anno_id_BPO.json","r") as f:
    anno_id_BPO=json.load(f)
with open("./Can_anc_files/id_candidate.json","r") as f:
    id_candidiate=json.load(f)
go_ancestor_num={}
for key in go_ancestor_BPO.keys():
    go_ancestor_num[key]=len(go_ancestor_BPO[key])
ancestors=[]
for key in go_ancestor_BPO.keys():
    for GO_id in go_ancestor_BPO[key]:
        ancestors.append(GO_id)
max_shot=8
ancestors=list(set(ancestors))
go_ancestor_BPO_new={}
for key in go_ancestor_BPO.keys():
    if key not in ancestors:
        go_ancestor_BPO_new[key]=go_ancestor_BPO[key]
leaf_keys=list(go_ancestor_BPO_new.keys())
model=torch.load("./output_model/12_1000_model.pt")
results=[]
for id in id_seq_test.keys():
    seq=id_seq_test[id]
    candidates=id_candidiate[id]   ###candidates: GO:1,GO:2......
    result=[]
    for candidate in candidates:
        candidate_ids=anno_id[candidate]
        if len(candidate_ids) > max_shot:
            candidate_ids=random.choices(candidates_ids,k=max_shot)
        candidate_seqs=[]
        for id_can in candidate_ids:
            candidate_seqs.append(id_seq_train[id_can])
        sim=float(model(seq,candidate_seqs))

        candidate_anc=go_ancestor_BPO[candidate]
        candidate_anc_num={}
        for anc in candidate_anc:
            candidate_anc_num[anc] = go_ancestor_num[anc]
        candidate_anc_num=sorted(candidate_anc_num.items(),key=lambda x:x[1])
        for i in range(math.ceil(len(candidate_anc)*sim)):
            if candidate_anc_num[i][0] not in result:
                result.append(candidate_anc_num[i][0])
    results.append(result)
print(results)
