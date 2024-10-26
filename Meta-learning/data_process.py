import pandas as pd
from obo_process import get_pandas
import json
ancestors_table=get_pandas()
def get_ancestor(id):
    indexes=[id]
    ancestor_indexes_total=[]

    while len(indexes)!=0:
        indexes_temp=[]
        for index in indexes:
            ancestor_indexes=ancestors_table[(ancestors_table.GO_id1==index)].index.to_list()
            for ancestor_index in ancestor_indexes:
                data=dict(ancestors_table.loc[ancestor_index])
                if data["relationship"]!="is a" and data["relationship"]!="part_of":
                    continue
                ancestor_indexes_total.append(data["GO_id2"])
                indexes_temp.append(data["GO_id2"])
        indexes=indexes_temp
    ancestor_indexes_total=set(ancestor_indexes_total)
    ancestor_indexes_total=list(ancestor_indexes_total)
    return ancestor_indexes_total

fasta_path="./finetune_data/data_heal/PDB_train_sequences.fasta"   ###The file path to be edited
anno_path="./finetune_data/data_heal/PDB_annot.tsv"        ###The file path to be edited
def get_data(fasta_path ,anno_path ,type_func="BPO"):
    id_seq ={}
    id_anno ={}
    anno_id ={}
    annos_total = pd.read_csv(anno_path, sep="\t", skiprows=12)
    with open(fasta_path ,'r') as f:
        pro_seq = ""
        num = 0
        pro_id_before = ""
        num =0

        for line in f:

            if ">" in line:
                if num % 2000 == 0:
                    print(num)
                num += 1
                
                if len(pro_seq) == 0:
                    pro_id_before = line.split(" ")[0][1:].strip()
                    continue
                if type_func == "BPO":
                    annos_key = list \
                        (annos_total[(annos_total["### PDB-chain"] == pro_id_before)]["GO-terms (biological_process)"])
                    if pd.isnull(annos_key[0]):

                        pro_id_before =line.split(" ")[0][1:].strip()
                        continue
                    annos_key =annos_key[0].split(",")
                    id_anno[pro_id_before ] =annos_key
                    for anno in annos_key:
                        if anno not in anno_id:
                            anno_id[anno ] =[pro_id_before]
                        else:
                            anno_id[anno].append(pro_id_before)
                    id_seq[pro_id_before] = pro_seq
                    pro_seq = ""
                    pro_id_before = line.split(" ")[0][1:].strip()
                    continue
            pro_seq += line.strip()


    return id_seq ,id_anno ,anno_id
print("start processing")
id_seq,id_anno,anno_id=get_data(fasta_path ,anno_path)
id_anno_new={}
anno_id_new={}
num=0
for key in id_seq:
    num+=1
    if num%100==0:
        print(str(num)+"/"+str(len(id_seq)))
    annos=id_anno[key]
    annos_label=[1 for _ in range(len(annos))]
    id_anno_new[key]=[]
    for anno in annos:
        anno_ancestors=get_ancestor(anno)
        for ancestor in anno_ancestors:
            if ancestor in annos:
                index=annos.index(ancestor)
                annos_label[index]=0
    for i in range(len(annos)):
        if annos_label[i]==1:
            id_anno_new[key].append(annos[i])
num=0
for key in id_anno:
    num+=1
    if num % 100 == 0:
        print(str(num) + "/" + str(len(id_seq)))
    annos=id_anno[key]
    for anno in annos:
        if anno not in anno_id_new:
            anno_id_new[anno]=[key]
        else:
            anno_id_new[anno].append(key)
go_ancestor={}
for key in anno_id:
    if key not in go_ancestor:
        anno_ancestors=get_ancestor(key)
        go_ancestor[key]=anno_ancestors
with open("./finetune_process_PDB/id_anno_BPO.json","w") as f:     ###The file path to be edited
    json.dump(id_anno_new,f)
with open("./finetune_process_PDB/anno_id_BPO.json","w") as f:      ###The file path to be edited
    json.dump(anno_id_new,f)
with open("./finetune_process_PDB/go_ancestor_BPO.json","w") as f:   ###The file path to be edited
    json.dump(go_ancestor,f)





