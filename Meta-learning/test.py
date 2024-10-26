import requests
import json
#r=requests.get("https://www.ebi.ac.uk/QuickGO/services/ontology/go/terms/GO%3A0042308/ancestors?relations=is_a%2Cpart_of%2Coccurs_in%2Cregulates")
#print(r.text)
import pandas as pd
import torch
import os

a=torch.tensor([[1,2,3,4],[3,4,5,6],[7,8,9,10],[3,4,3,3]])
b=a[2]
a=torch.cat((a[:2],a[3:]),dim=0)
print(b)
print("======================")
print(a)