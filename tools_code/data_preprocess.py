import numpy as np
from scipy.spatial.transform import Rotation as R
import json
import math
import cv2
import os
def get_poses(scene, root):
    """
    :param scene: Index of trajectory
    :param root: Root folder of dataset
    :return: all camera poses as quaternion vector and 4x4 projection matrix
    poss：the 3 position and 4 quaternion
    poses_mat: the matrix of rotation and position
    """
    locations = []
    rotations = []


    intri_reader = open(root + 'Frames_S' + str(scene) + '/cam.txt', 'r')
    loc_reader = open(root + 'SavedPosition_S' + str(scene) + '.txt', 'r')
    rot_reader = open(root + 'SavedRotationQuaternion_S' + str(scene) + '.txt', 'r')
    for line in loc_reader:
        locations.append(list(map(float, line.split())))

    for line in rot_reader:
        rotations.append(list(map(float, line.split())))


    locations = np.array(locations)
    rotations = np.array(rotations)
    poses = np.concatenate([locations, rotations], 1)

    r = R.from_quat(rotations).as_matrix()  #从四元数得到旋转矩阵

    TM = np.eye(4)  #对角线数组，
    TM[1, 1] = -1

    poses_mat = []
    for i in range(locations.shape[0]):
        ri = r[i]
        Pi = np.concatenate((ri, locations[i].reshape((3, 1))), 1)
        Pi = np.concatenate((Pi, np.array([0.0, 0.0, 0.0, 1.0]).reshape((1, 4))), 0)
        Pi_left = TM @ Pi @ TM   # Translate between left and right handed systems
        poses_mat.append(Pi_left)

# 内参
    intris = intri_reader.readlines()
    for content in intris:
        content1 = content.split()
        fx = content1[0]
        fy = content1[2]
        cx = content1[4]
        cy = content1[5]
#训练集，测试集，验证集分开  val 第一张 100张测试集，其余的训练集
    num=poses.shape[0]
    num_val = 100
    all_ids = np.arange(0, num)
    step = num // num_val
    test_ids = all_ids[1::step]
    val_ids = [0]
    train_ids = sorted(set(all_ids) - set(test_ids) - set(val_ids))
#写数据
    #训练集
    data_all=[]
    filename="transforms_"+"train.json"
    for i in train_ids:
        n=str(i).zfill(4)
        path= 'Frames_S' + str(scene) + '/FrameBuffer_'+str(n)
        data_detail={"file_path":path,
                     "rotation": 0.012566370614359171,
                     "transform_matrix":poses_mat[i].tolist()}
        data_all.append(data_detail)
        path2=root+path+'.png'
        save_path=root+'train'+'/FrameBuffer_'+str(n)+'.png'
        img=cv2.imread(path2)
        cv2.imwrite(save_path,img)

    w = 2 * float(cx)
    camera_angle_x = math.atan(w / (float(fx) * 2)) * 2
    dict_data = {"camera_angle_x": float(camera_angle_x),"frames":data_all}
    FileOperate(dictData=dict_data, filepath=root,
                filename=filename).operation_file()
    # 测试集
    data_all = []
    filename = "transforms_" + 'test.json'
    for i in test_ids:
        n = str(i).zfill(4)
        path = 'Frames_S' + str(scene) + '/FrameBuffer_' + str(n)
        data_detail = {"file_path": path,
                       "rotation": 0.012566370614359171,
                       "transform_matrix": poses_mat[i].tolist()}
        data_all.append(data_detail)
        path2 = root + path + '.png'
        save_path = root + 'test' + '/FrameBuffer_' + str(n) + '.png'
        img = cv2.imread(path2)
        cv2.imwrite(save_path, img)

    w = 2 * float(cx)
    camera_angle_x = math.atan(w / (float(fx) * 2)) * 2
    dict_data = {"camera_angle_x": float(camera_angle_x), "frames": data_all}
    FileOperate(dictData=dict_data, filepath=root,
                filename=filename).operation_file()
    # 验证集
    data_all = []
    filename = "transforms_" + 'val.json'
    for i in val_ids:
        n = str(i).zfill(4)
        path = 'Frames_S' + str(scene) + '/FrameBuffer_' + str(n)
        data_detail = {"file_path": path,
                       "rotation": 0.012566370614359171,
                       "transform_matrix": poses_mat[i].tolist()}
        data_all.append(data_detail)
        path2 = root + path + '.png'
        save_path = root + 'val' + '/FrameBuffer_' + str(n) + '.png'
        img = cv2.imread(path2)
        cv2.imwrite(save_path, img)
    w = 2 * float(cx)
    camera_angle_x = math.atan(w / (float(fx) * 2)) * 2
    dict_data = {"camera_angle_x": float(camera_angle_x), 'frames': data_all}
    FileOperate(dictData=dict_data, filepath=root,
                filename=filename).operation_file()
    return poses, np.array(poses_mat)

def get_diff_poses(scene,root):
    """
    :param scene: Index of trajectory
    :param root: Root folder of dataset
    :return: all camera poses as quaternion vector and 4x4 projection matrix
    poss：the 3 position and 4 quaternion
    poses_mat: the matrix of rotation and position
    """
    filename_ini="transforms.json"
    file_path=root+filename_ini
    with open(file_path, encoding='utf-8') as a:
        # 读取文件
        result = json.load(a)
        # 定义一个空数组
        new_list = []
        # 循环遍历
        camera_angle_x=result.get("camera_angle_x")
        print(camera_angle_x)
        frames=result.get("frames")
      #  frames=np.array(frames)

        frames_train=[]
        frames_test=[]
        frames_val=[]
        num=len(frames)
        num_val=10
        all_ids = np.arange(0, num)
        step = num // num_val
        test_ids = all_ids[1::step]
        val_ids = [0]
        train_ids = sorted(set(all_ids) - set(test_ids) - set(val_ids))
        print(num)
        for i in train_ids:
            frames_train.append(frames[i])
            # "./images/0.jpg"
            file_path=frames[i].get("file_path")
            save_path = root +'/train'
            if not os.path.exists(save_path):  # 如果路径不存在
                os.makedirs(save_path)
            save_path1 = save_path +"/", str(i) + '.png'

            img=cv2.imread(file_path)
           # print(img)
            print(save_path1)
            cv2.imwrite(save_path1,img)
            break
        for i in test_ids:
            frames_test.append(frames[i])
        for i in val_ids:
            frames_val.append(frames[i])


        filename="transforms_train.json"
        dict_data = {"camera_angle_x": float(camera_angle_x), "frames": frames_train}
        FileOperate(dictData=dict_data, filepath=root,
                         filename=filename).operation_file()

        filename = "transforms_test.json"
        dict_data = {"camera_angle_x": float(camera_angle_x), "frames": frames_test}
        FileOperate(dictData=dict_data, filepath=root,
                    filename=filename).operation_file()

        filename = "transforms_val.json"
        dict_data = {"camera_angle_x": float(camera_angle_x), "frames": frames_val}
        FileOperate(dictData=dict_data, filepath=root,
                    filename=filename).operation_file()

#         for i in result:  # i是个字典
#             # (i.get('name'), i.get('like'), i.get('address'))
#             # 熊猫 听歌 上海
#             # 老虎 运动 北京
#             new_list.append((i.get('name'), i.get('like'), i.get('address')))  # 将获取的值存入数组中
#         print(new_list)  # [('熊猫', '听歌', '上海'), ('老虎', '运动', '北京')]
#
#     #训练集，测试集，验证集分开  val 第一张 100张测试集，其余的训练集
#     num=poses.shape[0]
#     num_val = 100
#     all_ids = np.arange(0, num)
#     step = num // num_val
#     test_ids = all_ids[1::step]
#     val_ids = [0]
#     train_ids = sorted(set(all_ids) - set(test_ids) - set(val_ids))
# #写数据
#     #训练集
#     data_all=[]
#     filename="transforms_"+"train.json"
#     for i in train_ids:
#         n=str(i).zfill(4)
#         path= 'Frames_S' + str(scene) + '/FrameBuffer_'+str(n)
#         data_detail={"file_path":path,
#                      "rotation": 0.012566370614359171,
#                      "transform_matrix":poses_mat[i].tolist()}
#         data_all.append(data_detail)
#         path2=root+path+'.png'
#         save_path=root+'train'+'/FrameBuffer_'+str(n)+'.png'
#         img=cv2.imread(path2)
#         cv2.imwrite(save_path,img)
#
#     w = 2 * float(cx)
#     camera_angle_x = math.atan(w / (float(fx) * 2)) * 2
#     dict_data = {"camera_angle_x": float(camera_angle_x),"frames":data_all}
#     FileOperate(dictData=dict_data, filepath=root,
#                 filename=filename).operation_file()
#     # 测试集
#     data_all = []
#     filename = "transforms_" + 'test.json'
#     for i in test_ids:
#         n = str(i).zfill(4)
#         path = 'Frames_S' + str(scene) + '/FrameBuffer_' + str(n)
#         data_detail = {"file_path": path,
#                        "rotation": 0.012566370614359171,
#                        "transform_matrix": poses_mat[i].tolist()}
#         data_all.append(data_detail)
#         path2 = root + path + '.png'
#         save_path = root + 'test' + '/FrameBuffer_' + str(n) + '.png'
#         img = cv2.imread(path2)
#         cv2.imwrite(save_path, img)
#
#     w = 2 * float(cx)
#     camera_angle_x = math.atan(w / (float(fx) * 2)) * 2
#     dict_data = {"camera_angle_x": float(camera_angle_x), "frames": data_all}
#     FileOperate(dictData=dict_data, filepath=root,
#                 filename=filename).operation_file()
#     # 验证集
#     data_all = []
#     filename = "transforms_" + 'val.json'
#     for i in val_ids:
#         n = str(i).zfill(4)
#         path = 'Frames_S' + str(scene) + '/FrameBuffer_' + str(n)
#         data_detail = {"file_path": path,
#                        "rotation": 0.012566370614359171,
#                        "transform_matrix": poses_mat[i].tolist()}
#         data_all.append(data_detail)
#         path2 = root + path + '.png'
#         save_path = root + 'val' + '/FrameBuffer_' + str(n) + '.png'
#         img = cv2.imread(path2)
#         cv2.imwrite(save_path, img)
#     w = 2 * float(cx)
#     camera_angle_x = math.atan(w / (float(fx) * 2)) * 2
#     dict_data = {"camera_angle_x": float(camera_angle_x), 'frames': data_all}
#     FileOperate(dictData=dict_data, filepath=root,
#                 filename=filename).operation_file()
#     return poses, np.array(poses_mat)
def get_intrinsics(scene, root):
    """
    :param scene: Index of trajectory
    :param root: Root folder of dataset
    :return: fx,fy,cx,cy
    """
    intri_reader = open(root + 'Frames_S' + str(scene) + '/cam.txt', 'r')
    intris=intri_reader.readlines()
    for content in intris:
        content1 = content.split()
        fx = content1[0]
        fy = content1[2]
        cx = content1[4]
        cy = content1[5]

    return fx,fy,cx,cy

def get_relative_pose(pose_t0, pose_t1):
    """
    :param pose_tx: 4x4 camera pose describing camera to world frame projection of camera x.
    :return: Position of camera 1's origin in camera 0's frame.
    """
    return np.matmul(np.linalg.inv(pose_t0), pose_t1)

class FileOperate:
    '''
    需要传入文件所在目录，完整文件名。
    默认为只读，并将json文件转换为字典类型输出
    若为写入，需向dictData传入字典类型数据
    默认为utf-8格式
    '''
    def __init__(self,filepath,filename,way='r',dictData = None,encoding='utf-8'):
        self.filepath = filepath
        self.filename = filename
        self.way = way
        self.dictData = dictData
        self.encoding = encoding

    def operation_file(self):
        if self.dictData:
            self.way = 'w'
        with open(self.filepath + self.filename, self.way, encoding=self.encoding, newline='\n') as f:

            if self.dictData:
                data = json.dumps(self.dictData, indent=1)
                #print(data)
                f.write(data)
            else:
                if '.json' in self.filename:
                    data = json.loads(f.read())
                else:
                    data = f.read()
                return data


#pose,poses_mat=get_poses(1,"./nerf_synthetic/medical/")
get_diff_poses('images',"/Users/jiawei/yufei/nerf-pytorch/data/nerf_synthetic/shi/")






