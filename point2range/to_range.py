import numpy as np
import matplotlib.pyplot as plt

file_path = './kitti.pcd'
npy_file_name = './pcd'

image_rows_full = 64  #行数
image_cols = 1024     #列数

# Ouster OS1-64 (gen1)
ang_res_x = 360.0/float(image_cols) # 水平个像素所对应的度数 resolution
ang_res_y = 33.2/float(image_rows_full-1) #垂直方向上每个像素对应的度数vertical resolution
ang_start_y = 16.6                        # 底部开始识别的角度beam angle
max_range = 80.0
min_range = 2.0

def load_pcd_data(file_path):
    pts = []
    f = open(file_path, 'r')
    data = f.readlines()#方法用于读取所有行(直到结束符 EOF)并返回列表，该列表可以由 Python 的 for... in ... 结构进行处理。
    f.close()
    line = data[9]
    # print line
    line = line.strip('\n')
    print (line)
    i = line.split(' ')
    print('111',i)
    pts_num = eval(i[-1])
    for line in data[11:]:
        line = line.strip('\n')
        xyzi = line.split(' ')
        x, y, z = [eval(i) for i in xyzi[:3]]
# 		# print type(bgra)
# 		argb = bin(eval(argb))[2:]
# 		a, r, g, b = [int(argb[8 * i:8 * i + 8], 2) for i in range(4)]
        pts.append([x, y, z])
    assert len(pts) == pts_num
    points_obj= np.zeros((pts_num, len(pts[0])), dtype=np.float)
    for i in range(pts_num):
        points_obj[i] = pts[i]
	# x = np.zeros([np.array(t) for t in pts])
    return points_obj

def pcd2range(file_path):
    print('#'*50)
    print('Dataset name: {}'.format(file_path))
    range_image_array = np.empty([0, image_rows_full, image_cols, 1], dtype=np.float32)
    points_obj = load_pcd_data(file_path)
    points_array = np.array(list(points_obj), dtype=np.float32)
    #print(points_array)
    # project points to range image转换成距离图像
    range_image = np.zeros((1, image_rows_full, image_cols, 1), dtype=np.float32)
    x = points_array[:,0]#取所有行中的第0个数据
    y = points_array[:,1]
    z = points_array[:,2]
    # find row id计算行id  绕x轴旋转角度加起始角度 最后除去每个像素所对应的角度并取整
    vertical_angle = np.arctan2(z, np.sqrt(x * x + y * y)) * 180.0 / np.pi
    relative_vertical_angle = vertical_angle + ang_start_y
    rowId = np.int_(np.round_(relative_vertical_angle / ang_res_y))
    # find column id  -[(绕z轴旋转角度-90)/每个像素对应的度数]+1024/2,相当于从后面截开再翻转
    horitontal_angle = np.arctan2(x, y) * 180.0 / np.pi
    colId = -np.int_((horitontal_angle-90.0)/ang_res_x) + image_cols/2;
    shift_ids = np.where(colId>=image_cols)
    colId[shift_ids] = colId[shift_ids] - image_cols
    # filter range过滤距离
    thisRange = np.sqrt(x * x + y * y + z * z)
    thisRange[thisRange > max_range] = 0
    thisRange[thisRange < min_range] = 0
    # save range info to range image储存信息到距离图像
    for i in range(len(thisRange)):
#      print(1111111111,i,len(thisRange),rowId[i],thisRange[i],colId[i])
        if rowId[i] < 0 or rowId[i] >= image_rows_full or colId[i] < 0 or colId[i] >= image_cols:
            continue
        range_image[0, rowId[i], int(colId[i]), 0] = thisRange[i]
    print('finish')
    # append range image to array
    range_image_array = np.append(range_image_array, range_image, axis=0)

    # save full resolution image array储存高分辨率图像数组
    np.save(npy_file_name, range_image_array)
    print('Dataset saved: {}'.format(npy_file_name))


pcd2range(file_path)

image3 = np.load("./pcd.npy")
print(image3.shape)
plt.imshow(image3[0,:,:,0])
#cv2.imwrite('/home/idnihai/lc/fuxian/yuce/input/'+str(i)+".png",image3[i,:,:])
plt.show()
