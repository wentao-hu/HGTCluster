# ## generate fake node data locally

# %%
import numpy as np
import pandas as pd
from random import choice
import random
import os
import sys

user_ids = list(range(1000))
note_ids = list(range(1000,2000))

# %%
user_list = []
for id in user_ids:
    age = random.randint(10,50)
    user_data = (id,age)
    user_list.append(user_data)

# %%
note_list=[]
for id in note_ids:
    views = random.randint(200,1000)
    data = (id,views)
    note_list.append(data)

# %%
current_path = sys.path[0]
user_node_path = os.path.abspath(os.path.join(current_path,'..'))+'/data/local/fakeuser'
if not os.path.exists(user_node_path):
    os.makedirs(user_node_path)

with open('{}/fakeuser.csv'.format(user_node_path),"w") as pos_f:
        pos_f.write('id:int64'+'\t'+'feature:string'+'\n')
with open('{}/fakeuser.csv'.format(user_node_path),"a") as pos_f:
    for line in user_list:
        pos_f.write(str(line[0])+'\t'+str(line[1]))
        pos_f.write('\n')



# %%
current_path = sys.path[0]
note_node_path = os.path.abspath(os.path.join(current_path,'..'))+'/data/local/fakenote'
if not os.path.exists(note_node_path):
    os.makedirs(note_node_path)

with open('{}/fakenote.csv'.format(note_node_path),"w") as pos_f:
    pos_f.write('id:int64'+'\t'+'feature:string'+'\n')
with open('{}/fakenote.csv'.format(note_node_path),"a") as pos_f:
    for line in note_list:
        pos_f.write(str(line[0])+'\t'+str(line[1]))
        pos_f.write('\n')


