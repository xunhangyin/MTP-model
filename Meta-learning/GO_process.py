import pandas as pd
import requests
import json
from obo_process import get_pandas

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
'''

    #for index in indexes:
    #    ancestores.append(ancestors_table.loc[index]["GO_id2"])

for i in range(len(terms_id)):
    terms_id[i]=terms[i][3:]
labels=[1 for _ in range(len(terms))]

#url_part1="https://www.ebi.ac.uk/QuickGO/services/ontology/go/terms/GO%3A"
#url_part2="/ancestors?relations=is_a%2Cpart_of%2Coccurs_in"
#def get_ancestor(id):
#see=ancestors_table[(ancestors_table.GO_id1=="GO:0043207")].index.to_list()
#print(ancestors_table.loc[see])
for i in range(len(terms)):
    print(i)
    if labels[i]==0:
        print("skip")
        continue
    else:
        #url=url_part1+terms_id[i]+url_part2
        #response=requests.get(url).json()

        #if "ancestors" not in response["results"][0]:
        #    labels[i]=0
        #    continue
        ancestors=get_ancestor(terms[i])
        for ancestor in ancestors:
            if ancestor==terms[i]:
                continue
            if terms.count(ancestor)==0:
                index=-1
            else:
                index=terms.index(ancestor)

            if index!=-1:
                labels[index]=0
one_index=[]
print(labels)
print(terms)
for i in range(len(labels)):
    if labels[i]==1:
        one_index.append(i)
for index in one_index:
    leaf_id=terms_id[index]

    #response=requests.get(url_part1+leaf_id+url_part2).json()
    #if response["results"][0]["isObsolete"] ==True:
    #    print("not use")
    #    continue
    print(terms[index])

    leaf_ancestors=get_ancestor(terms[index])
    for l_ance in leaf_ancestors:
        
        if l_ance not in terms:

            print(l_ance)
            print("not in")
    print("========================================")'''
