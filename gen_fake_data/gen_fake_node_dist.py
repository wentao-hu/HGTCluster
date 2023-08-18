# ## generate fake node data distributionally

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
unit = len(user_list)//20
print('user number of each part: ',unit)
current_path = sys.path[0]
user_node_path = os.path.abspath(os.path.join(current_path,'..'))+'/data/dist/fakeuser'
if not os.path.exists(user_node_path):
    os.makedirs(user_node_path)

for i in range(20):
    random.shuffle(user_list)
    part_i= user_list[i*unit:(i+1)*unit]
   
    with open('{}/part-{}.csv'.format(user_node_path, i),"w") as pos_f:
        pos_f.write('id:int64'+'\t'+'feature:string'+'\n')
    with open('{}/part-{}.csv'.format(user_node_path, i),"a") as pos_f:
        for line in part_i:
            pos_f.write(str(line[0])+'\t'+str(line[1]))
            pos_f.write('\n')

# %%
unit = len(note_list)//20
print('note number of each part: ',unit)
current_path = sys.path[0]
note_node_path = os.path.abspath(os.path.join(current_path,'..'))+'/data/dist/fakenote'
if not os.path.exists(note_node_path):
    os.makedirs(note_node_path)

for i in range(20):
    random.shuffle(note_list)
    part_i= note_list[i*unit:(i+1)*unit]
   
    with open('{}/part-{}.csv'.format(note_node_path,i),"w") as pos_f:
        pos_f.write('id:int64'+'\t'+'feature:string'+'\n')
    with open('{}/part-{}.csv'.format(note_node_path,i),"a") as pos_f:
        for line in part_i:
            pos_f.write(str(line[0])+'\t'+str(line[1]))
            pos_f.write('\n')


