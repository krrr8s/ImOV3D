import os
import numpy as np

type2class = {
                            "toilet": 0,
                            "bed": 1,
                            "chair": 2,
                            "sofa": 3,
                            "dresser": 4,
                            "table": 5,
                            "cabinet": 6,
                            "bookshelf": 7,
                            "pillow": 8,
                            "sink": 9,
                            "bathtub": 10,
                            "refridgerator": 11,
                            "desk": 12,
                            "night stand": 13,
                            "counter": 14,
                            "door": 15,
                            "curtain": 16,
                            "box": 17,
                            "lamp": 18,
                            "bag": 19}

class2type = {type2class[t]: t for t in type2class}
def get_box3d_dim_statistics(folder_path):
    """ Collect 3D bounding box statistics.
    Used for computing mean box sizes. """
    dimension_list = []
    type_list = []
    ry_list = []
    data_idx_list = []
    for file in os.listdir(folder_path):
        if file.endswith("_bbox.npy") and not file.endswith("_2d_bbox.npy"):
            print(file)
            data_idx_list.append(file)
    print("data_idx_list", data_idx_list)
    for data_idx in data_idx_list:
        print('------------- ', data_idx)
        # 构建完整的文件路径
        data_file_path = os.path.join(folder_path, data_idx)
        # 使用np.load加载.npy文件
        objects = np.load(data_file_path)
        for obj_idx in range(len(objects)):
            obj = objects[obj_idx]
            dimension_list.append(np.array([obj[3], obj[4], obj[5]]))
            type_list.append(class2type[int(obj[-1])])

            # Get average box size for different catgories
    for class_type in sorted(set(type_list)):
        cnt = 0
        box3d_list = []
        for i in range(len(dimension_list)):
            if type_list[i] == class_type:
                cnt += 1
                box3d_list.append(dimension_list[i])
        median_box3d = np.median(box3d_list, 0)
        print("\'%s\': np.array([%f,%f,%f])," % \
              (class_type, median_box3d[0] * 2, median_box3d[1] * 2, median_box3d[2] * 2))


folder_path = "/share/timingyang/OV-3DET/Data_Maker/Test_GT_Maker/scannet_frames_25k_sunformat_val"
get_box3d_dim_statistics(folder_path)