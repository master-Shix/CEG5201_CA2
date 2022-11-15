import json
import os


originalData = "/home/jiawei/weijia/literature/ourModels/ourbaseline_data/flower_3Views"
# split = ['transforms_train.json','transforms_test.json','transforms_val.json']
views = 3
dic = {}
with open(os.path.join(originalData,"transforms_train.json"), 'r') as load_f:
    load_dict = json.load(load_f)
    num = 0
    for frame in load_dict['frames']:
        path = frame['file_path']
        dic[path] = {}
        if num<100:
            dic[path]['time_id'] = 0
            dic[path]['warp_id'] = 0
            dic[path]['appearance_id'] = 0
            dic[path]['camera_id'] = 0
        else:
            dic[path]['time_id'] = (num-100)//views
            dic[path]['warp_id'] = (num-100)//views
            dic[path]['appearance_id'] = (num-100)//views
            dic[path]['camera_id'] = 0
        num+=1


views = 1

with open(os.path.join(originalData,"transforms_test.json"), 'r') as load_f:
    load_dict = json.load(load_f)
    num = 0
    for frame in load_dict['frames']:
        path = frame['file_path']
        dic[path] = {}
        dic[path]['time_id'] = (num)//views
        dic[path]['warp_id'] = (num)//views
        dic[path]['appearance_id'] = (num)//views
        dic[path]['camera_id'] = 0
        num+=1

# with open(os.path.join(originalData,"transforms_val.json"), 'r') as load_f:
#     load_dict = json.load(load_f)
#     num = 0
#     for frame in load_dict['frames']:
#         path = frame['file_path']
#         dic[path] = {}
#         if num<100:
#             dic[path]['time_id'] = 0
#             dic[path]['warp_id'] = 0
#             dic[path]['appearance_id'] = 0
#             dic[path]['camera_id'] = 0
#         else:
#             # path = frame['file_path']
#             # n = int(path.split("_")[-1])
#             # n = n//50+views
#             dic[path]['time_id'] = (num-100)//views
#             dic[path]['warp_id'] = (num-100)//views
#             dic[path]['appearance_id'] = (num-100)//views
#             dic[path]['camera_id'] = 0
#         num+=1

f2 = os.path.join(originalData,"metadata.json")
with open(f2,"w") as dump_f:
    json.dump(dic,dump_f,indent=4)
