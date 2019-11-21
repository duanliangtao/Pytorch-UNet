import SimpleITK as sitk
import numpy as np
import csv
import os
from glob import glob

import matplotlib
import matplotlib.pyplot as plt

import pandas as pd
try:
    from tqdm import tqdm # 进度条展示包
except:
    print('TQDM does make much nicer wait bars...')
    tqdm = lambda x: x

file_folder = 'D:/AI-test/test/mhd/'
luna_path = 'D:/AI-test/test/csv/'
output_path = "D:/AI-test/test/output/"
files = [];




# 生成标注图
def make_mask(center,diam,z,width,height,spacing,origin):
    '''根据CT坐标下的结节标注信息，返回真实坐标下的结节mask
    Center : 结节标注坐标x,y,z
    diam : 结节标注直径,mm
    width,height : 图片尺寸
    spacing： x,y,z方向的CT像素间隔，mm
    origin： x,y,z CT原点
    z：真实切片对应的CT坐标系下的z坐标
    '''
    mask = np.zeros([height,width]) # 0's everywhere except nodule swapping x,y to match img

    # 获取结节真实坐标的半径左下右上坐标，此处将结节区域扩大10像素？
    v_center = (center-origin)/spacing # 结节真实中心
    v_diam = int(diam/spacing[0]+5) # 结节真实半径+5
    v_xmin = np.max([0,int(v_center[0]-v_diam)-5])  # 结节真实x中心+ 结节真实半径-5
    v_xmax = np.min([width-1,int(v_center[0]+v_diam)+5]) # 结节真实x中心+结节真实半径+5
    v_ymin = np.max([0,int(v_center[1]-v_diam)-5])  # 结节真实y中心+ 结节真实半径-5
    v_ymax = np.min([height-1,int(v_center[1]+v_diam)+5])  # 结节真实y中心+ 结节真实半径+5
    #print(v_diam,v_xmin,v_xmax,v_ymin,v_ymax)

    v_xrange = range(v_xmin,v_xmax+1)
    v_yrange = range(v_ymin,v_ymax+1)

    # Convert back to world coordinates for distance calculation
    x_data = [x*spacing[0]+origin[0] for x in range(width)]
    y_data = [x*spacing[1]+origin[1] for x in range(height)]

    # 将有结节的圆形区域的值设置为1，其他设置为0
    for v_x in v_xrange:
        for v_y in v_yrange:
            p_x = spacing[0]*v_x + origin[0]
            p_y = spacing[1]*v_y + origin[1]
            if np.linalg.norm(center-np.array([p_x,p_y,z]))<=diam:
                # 注意mask坐标顺序与图片array坐标顺序匹配
                mask[int((p_y-origin[1])/spacing[1]),int((p_x-origin[0])/spacing[0])] = 1.0
    return(mask)

def get_all_mhd_files():
    filesList = []
    for home, dirs, files in os.walk(file_folder):
        for filename in files:
            # 文件名列表，包含完整路径
            if '.mhd' in filename:
                filesList.append(os.path.join(home, filename))
    return filesList
def get_filename(case):
    global files
    for f in files:
        if case in f:
            return(f)
def mask_mhd_files():
    df_node = pd.read_csv(luna_path + "annotations.csv")
    df_node["file"] = df_node["seriesuid"].apply(get_filename)
    df_node = df_node.dropna()  # 提取有肺结节标注的病历
    for fcount, img_file in enumerate(tqdm(files)):
        mini_df = df_node[df_node["file"] == img_file]  # get all nodules associate with file
        if mini_df.shape[0] > 0:  # some files may not have a nodule--skipping those
            # load the data once
            itk_img = sitk.ReadImage(img_file)
            img_array = sitk.GetArrayFromImage(itk_img)  # indexes are z,y,x (notice the ordering)
            print("image shape:", img_array.shape)
            num_z, height, width = img_array.shape  # heightXwidth constitute the transverse plane
            origin = np.array(itk_img.GetOrigin())  # x,y,z  Origin in world coordinates (mm)
            spacing = np.array(itk_img.GetSpacing())  # spacing of voxels in world coor. (mm)
            # go through all nodes (why just the biggest?)
            for node_idx, cur_row in mini_df.iterrows():
                node_x = cur_row["coordX"]
                node_y = cur_row["coordY"]
                node_z = cur_row["coordZ"]
                diam = cur_row["diameter_mm"]
                # 只获取离肺结节显示最大的三个CT切片
                imgs = np.ndarray([3, height, width], dtype=np.float32)
                masks = np.ndarray([3, height, width], dtype=np.uint8)
                center = np.array([node_x, node_y, node_z])  # nodule center
                v_center = np.rint((center - origin) / spacing)  # nodule center in voxel space (still x,y,z ordering)
                print("3 slice index:", np.arange(int(v_center[2]) - 1, int(v_center[2]) + 2).clip(0, num_z - 1))
                for i, i_z in enumerate(np.arange(int(v_center[2]) - 1,
                                                  int(v_center[2]) + 2).clip(0,
                                                                             num_z - 1)):  # clip prevents going out of bounds in Z
                    mask = make_mask(center, diam, i_z * spacing[2] + origin[2],
                                     width, height, spacing, origin)
                    masks[i] = mask
                    imgs[i] = img_array[i_z]
            file_name = os.path.splitext(os.path.basename(img_file))[0]
            for i in range(len(imgs)):
                matplotlib.image.imsave(output_path + '/img/'+file_name+'_' + str(i) + '.jpg', imgs[i], cmap='gray')
                matplotlib.image.imsave(output_path + '/mask/'+file_name+'_' + str(i) + '.jpg', masks[i], cmap='gray')

if __name__=="__main__":
    print("开始处理")
    # 读取文件夹下所有的mhd文件
    files = get_all_mhd_files()
    # 将mhd进行标注图生成
    mask_mhd_files()

    print("处理结束")

    exit()
