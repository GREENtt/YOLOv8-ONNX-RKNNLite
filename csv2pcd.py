import os
import numpy as np
import pandas as pd

filepath = 'C:/Users/lzy06/Desktop/zxq/RSView/lidar_img/csv'

PCD_DIR_PATH = 'C:/Users/lzy06/Desktop/zxq/RSView/lidar_img/board_lidar'

# 传入点云对象
def pointsTopcd(points, name):
    # 存放路径

    PCD_FILE_PATH = os.path.join(PCD_DIR_PATH, name + '.pcd')
    if os.path.exists(PCD_FILE_PATH):
        os.remove(PCD_FILE_PATH)

    # 写文件句柄
    handle = open(PCD_FILE_PATH, 'a')

    # 得到点云点数
    point_num = len(data)

    # pcd头部
    handle.write(
        '# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1')
    string = '\nWIDTH ' + str(point_num)
    handle.write(string)
    handle.write('\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0')
    string = '\nPOINTS ' + str(point_num)
    handle.write(string)
    handle.write('\nDATA ascii')

    # 依次写入点
    for i in range(point_num):  # 这里我只用到了前三列，故只需展示0，1，2三列 读者可根据自身需要写入其余列
        string = '\n' + str(points[i, 0]) + ' ' + str(points[i, 1]) + ' ' + str(points[i, 2])
        handle.write(string)
    handle.close()


if __name__ == '__main__':
    filenames = os.listdir(filepath)
    for filename in filenames:
        datapath = os.path.join(filepath, filename)
        data = pd.read_csv(datapath, encoding='utf-8')  # 读取csv文件
        data_234 = data.iloc[:, 3:6]  # 这里做的是切割，因为我这里只用到了其收集数据的第1列到第三列，即x,y,z 可根据自身需要设置
        data_234 = np.array(data_234)  # 转换为numpy方便计算
        data_new = data_234
        firstname = filename.split('.')[0]
        pointsTopcd(data_new, firstname)
'''

import warnings
import open3d as o3d
import numpy as np
import pandas as pd
import pyntcloud

path = '你的点云文件.csv'


# CSV转PCD
def read_point(file_path):
    pointcloud_data = np.asarray(pd.read_csv(file_path))
    return pointcloud_data


csv_data = read_point(path)
# csv文件的数据被存入数组中
# 提取点云坐标，所有行，第二列到第三列（根据自己的点云文件存储的信息决定读取哪几列）
points = csv_data[:, 1:4]

# 创建Open3D的PointCloud对象
pcd = o3d.geometry.PointCloud()

# 设置点云的坐标
pcd.points = o3d.utility.Vector3dVector(points)

# 变量pcd这时候已经是一个pcd文件了
# 保存为PCD文件
o3d.io.write_point_cloud('21.pcd', pcd)

warnings.filterwarnings("ignore")

path = '21.pcd'

# 读取PCD文件
cloud = pyntcloud.PyntCloud.from_file(path)

# pcd转ply
# 保存为PLY文件
cloud.to_file('21.ply', also_save=["mesh"])

# pcd转xyz
# 保存为XYZ文件
cloud.points.to_csv('output.xyz')

'''
