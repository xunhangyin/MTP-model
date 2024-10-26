import json
def get_data(train_fasta_path,test_fasta_path,func_type="BPO"):
    id_seq_train={}
    id_seq_test={}
    id_anno_train={}
    id_anno_test={}
    anno_id_train={}
    anno_id_test={}
    go_ancestor_train={}
    go_ancestor_test={}
    with open(train_fasta_path,"r") as f:
        pro_id_before=""
        pro_seq=""
        for line in f:
            if ">" in line:
                if len(pro_seq) == 0:
                    pro_id_before = line.split(" ")[0][1:].strip()
                    continue
                id_seq_train[pro_id_before] = pro_seq
                pro_seq = ""
                pro_id_before = line.split(" ")[0][1:].strip()
                continue
            pro_seq+=line.strip()
        f.close()
    with open(test_fasta_path,"r") as f:
        pro_id_before=""
        pro_seq=""
        for line in f:
            if ">" in line:
                if len(pro_seq) == 0:
                    pro_id_before = line.split(" ")[0][1:].strip()
                    continue
                id_seq_test[pro_id_before] = pro_seq
                pro_seq=""
                pro_id_before = line.split(" ")[0][1:].strip()
                continue
            pro_seq+=line.strip()
        f.close()
    with open("./finetune_process_PDB/id_anno_"+func_type+".json","r") as f:
        id_anno_train = json.load(f)
        f.close()
    with open("./finetune_process_PDB/anno_id_"+func_type+".json","r") as f:
        anno_id_train = json.load(f)
        f.close()
    with open("./finetune_process_PDB/go_ancestor_"+func_type+".json","r") as f:
        go_ancestor_train = json.load(f)
    with open("./finetune_process_PDB/id_anno_"+func_type+"_test.json","r") as f:
        id_anno_test = json.load(f)
        f.close()
    with open("./finetune_process_PDB/anno_id_"+func_type+"_test.json","r") as f:
        anno_id_test = json.load(f)
        f.close()
    with open("./finetune_process_PDB/go_ancestor_"+func_type+"_test.json","r") as f:
        go_ancestor_test = json.load(f)
        f.close()
    return id_seq_train,id_seq_test,id_anno_train,id_anno_test,anno_id_train,anno_id_test,go_ancestor_train,go_ancestor_test

