# ## generate fake edge

# %%
import numpy as np
import pandas as pd
from random import choice
import os 
import sys
user_ids = list(range(1000))
note_ids = list(range(1000,2000))

# %%
# 生成fake pos follow note 关系
follow_note = set()  # follow_note是一个二元组，不重复
for id in user_ids:
    neigh_count = 0
    max_neigh_count = np.random.randint(10,20)
    # if id%100==0:
    #    print(id,max_neigh_count) 
    while neigh_count<= max_neigh_count:   # 每个user在follow note中有100-200个正边
        note = np.random.randint(1000,2000)
        edge = (id, note)
        if edge not in follow_note:
            follow_note.add(edge)
            neigh_count+=1
# print('end of for genrating follow_note')
pos_follow_note_list = list(follow_note)
print(pos_follow_note_list[:20])
new_pos_list = sorted(pos_follow_note_list, key=lambda k: (k[0],k[1]))
print(new_pos_list[:20])

# %%
pos_unit = len(pos_follow_note_list)//20
print('{} follow note edges in each part '.format(pos_unit))
current_path = sys.path[0]
follow_note_path = os.path.abspath(os.path.join(current_path,'..'))+'/data/dist/fake_follow_note'
if not os.path.exists(follow_note_path):
    os.makedirs(follow_note_path)

for i in range(20):
    pos_part_i= pos_follow_note_list[i*pos_unit:(i+1)*pos_unit]
    with open('{}/part-{}.csv'.format(follow_note_path,i),"w") as pos_f:
        pos_f.write('src_id:int64'+'\t'+'dst_id:int64'+'\t'+'weight:float'+'\t'+'label:int32'+'\n')
    with open('{}/part-{}.csv'.format(follow_note_path,i),"a") as pos_f:
        for line in pos_part_i:
            pos_f.write(str(line[0])+'\t'+str(line[1])+'\t'+'1.0'+'\t'+'1')
            pos_f.write('\n')
    

# %%
share_note = set()  # share_note是一个二元组，不重复
for id in user_ids:
    neigh_count = 0
    max_neigh_count = np.random.randint(3,8)
    # if id%100==0:
    #    print(id,max_neigh_count) 
    while neigh_count<= max_neigh_count:   # share note中有3-8个正边
        note = np.random.randint(1000,2000)
        edge = (id, note)
        if edge not in follow_note:
            share_note.add(edge)
            neigh_count+=1
print('end of generating share_note')
pos_share_note_list = list(share_note)
print(pos_share_note_list[:20])
new_pos_list = sorted(pos_share_note_list, key=lambda k: (k[0],k[1]))
print(new_pos_list[:20])

# %%
pos_unit = len(pos_share_note_list)//20
print('{} share note edges in each part '.format(pos_unit))
current_path = sys.path[0]
share_note_path = os.path.abspath(os.path.join(current_path,'..'))+'/data/dist/fake_share_note'
if not os.path.exists(share_note_path):
    os.makedirs(share_note_path)
    
for i in range(20):
    pos_part_i= pos_share_note_list[i*pos_unit:(i+1)*pos_unit]
    with open('{}/part-{}.csv'.format(share_note_path,i),"w") as pos_f:
        pos_f.write('src_id:int64'+'\t'+'dst_id:int64'+'\t'+'weight:float'+'\t'+'label:int32'+'\n')
    with open('{}/part-{}.csv'.format(share_note_path,i),"a") as pos_f:
        for line in pos_part_i:
            pos_f.write(str(line[0])+'\t'+str(line[1])+'\t'+'1.0'+'\t'+'1')
            pos_f.write('\n')


