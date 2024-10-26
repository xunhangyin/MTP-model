import torch
from ESMpredictor import Esmpredictor
import argparse
from get_parse import parser_add_main_args
from GO_process import get_ancestor
import json
import random
from get_data_new import get_data
train_fasta_path="./finetune_data/data_heal/PDB_train_sequences.fasta"
test_fasta_path="./finetune_data/data_heal/PDB_test_sequences.fasta"
id_seq,id_seq_test,id_anno,id_anno_test,anno_id,anno_id_test,go_ancestor,go_ancestor_test=get_data(train_fasta_path,test_fasta_path)
predictor=Esmpredictor("./model_output/",load_step=10000,max_shot=64)
candidate_GO_leaf_terms=predictor.predict(id_seq,id_seq_test,anno_id,id_anno_test,go_ancestor,go_ancestor_test)

