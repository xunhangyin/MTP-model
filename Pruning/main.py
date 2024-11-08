import torch
from ESM_trainer import EsmTrainer
import argparse
from get_parse import parser_add_main_args
from GO_process import get_ancestor
import json
import random
from get_data_new import get_data
parser = argparse.ArgumentParser(description='General Training Pipeline')
parser_add_main_args(parser)
args = parser.parse_args()
leaves=[]
train_fasta_path="./finetune_data/data_heal/PDB_train_sequences.fasta"
test_fasta_path="./finetune_data/data_heal/PDB_test_sequences.fasta"
id_seq,id_seq_test,id_anno,id_anno_test,anno_id,anno_id_test,go_ancestor,go_ancestor_test=get_data(train_fasta_path,test_fasta_path)
with open("./finetune_process_PDB/go_ancestor_BPO.json","r") as f:
    go_ancestor_BPO=json.load(f)
with open("./finetune_process_PDB/anno_id_BPO.json","r") as f:
    anno_id_BPO=json.load(f)

ancestors=[]
for key in go_ancestor_BPO.keys():
    for GO_id in go_ancestor_BPO[key]:
        ancestors.append(GO_id)

ancestors=list(set(ancestors))
go_ancestor_BPO_new={}
for key in go_ancestor_BPO.keys():
    if key not in ancestors:
        go_ancestor_BPO_new[key]=go_ancestor_BPO[key]
leaf_keys=list(go_ancestor_BPO_new.keys())
anno_id={}
for key in id_anno:
    annos=id_anno[key]
    for anno in annos:
        if anno not in anno_id:
            anno_id[anno]=[key]
        else:
            anno_id[anno].append(key)
trainer=EsmTrainer(args.model_path,args.output_dir,args.lr,args.batch_size,args.epochs,args.weight_decay,args.accumulate_steps,eval_step=args.eval_step)
use_large=False

trainer.train(id_seq,id_anno,anno_id,id_seq_test,id_anno_test,anno_id_test,go_ancestor,go_ancestor_test,leaf_keys,use_large)


