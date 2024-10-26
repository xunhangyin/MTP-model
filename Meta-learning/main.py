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
train_fasta_path="./finetune_data/data_heal/PDB_train_sequences.fasta"  ###The file path to be edited
test_fasta_path="./finetune_data/data_heal/PDB_test_sequences.fasta"    ###The file path to be edited
id_seq,id_seq_test,id_anno,id_anno_test,anno_id,anno_id_test,go_ancestor,go_ancestor_test=get_data(train_fasta_path,test_fasta_path)
trainer=EsmTrainer(args.model_path,args.output_dir,args.lr,args.batch_size,args.epochs,args.weight_decay,args.accumulate_steps,eval_step=args.eval_step)
use_large=False
trainer.train(id_seq,id_anno,anno_id,id_seq_test,id_anno_test,anno_id_test,go_ancestor,go_ancestor_test,use_large)


