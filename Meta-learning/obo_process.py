from tqdm import tqdm
import pandas as pd
def get_pandas():
    f = open('./finetune_data/data_heal/go-basic.obo')
    lines = f.readlines()

    nodes1 = []
    names1 = []
    namespaces = []
    relationships = []
    nodes2 = []
    names2 = []


    def append_all_lists():
        nodes1.append(node1)
        names1.append(name1)
        namespaces.append(namespace)
        relationships.append(relationship)
        nodes2.append(node2)
        names2.append(name2)


    for line in tqdm(lines):
        line = line.strip()
        if line.find('id:') == 0:
            node1 = line[line.find('GO:'):].strip()
        elif line.find('name:') == 0:
            name1 = line[5:].strip()
        elif line.find('namespace:') == 0:
            namespace = line[10:].strip()
        elif line.find('is_a:') == 0:
            relationship = 'is a'
            node2 = line.split('!')[0][5:].strip()
            name2 = line.split('!')[1].strip()
            append_all_lists()
        elif line.find('relationship:') == 0:
            relationship = line.split('GO:')[0][13:].strip()
            node2 = line.split('!')[0][line.find('GO:'):].strip()
            name2 = line.split('!')[1].strip()
            append_all_lists()

    GO_GO_data = pd.DataFrame([nodes1, names1, namespaces, relationships, nodes2, names2]).T
    GO_GO_data.columns = ['GO_id1', 'GO_name1', 'type', 'relationship', 'GO_id2', 'GO_name2']
    return GO_GO_data
