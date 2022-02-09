from cmath import pi
import numpy as np
import math
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


point_file = './pose1_full.pcd'
img = './pose1.png'

#cam to lidar
roll = -1.51318
pitch = -0.00278
yaw = -1.47518
x = 0.97722
y = 1.33155
z = 0.52005
K = np.array([656.043,0.0,286.564,0.0,654.308,221.388,0.0,0.0,1])
K = K.reshape(3,3)

def rotmatrix(roll,pitch,yaw):
    rot_x = np.mat([[1,0,0],
                    [0,math.cos(roll),-math.sin(roll)],
                    [0,math.sin(roll),math.cos(roll)]])


    rot_y = np.mat([[math.cos(pitch),0,math.sin(pitch)],
                        [0,1,0],
                        [-math.sin(pitch),0,math.cos(pitch)]]) 

    rot_z = np.mat([[math.cos(yaw),-math.sin(yaw),0],
                        [math.sin(yaw),math.cos(yaw),0],
                        [0,0,1]])
    rot1 = np.matmul(rot_z , rot_y)
    rot = np.matmul(rot1 ,rot_x)
    print("rot_matrix:",rot)
    return(rot)

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
    points_obj= np.zeros((pts_num, len(pts[0])), dtype=np.float64)
    for i in range(pts_num):
        points_obj[i] = pts[i]
    points_obj = points_obj
    print(points_obj.shape)
	# x = np.zeros([np.array(t) for t in pts])
    return points_obj

move = np.array([[x],[y],[z]])
rot = rotmatrix(roll,pitch,yaw)
rot1 = rot.T
print('translation',move,'rot',rot1)





#translation
points = load_pcd_data(point_file).T
points1 = points - move

points1 = np.delete(points1,np.where(points1[0,:]<0),axis=1)

#rotation
points2 = np.matmul(rot1,points1)

#K
cam = np.matmul(K,points2)
cam = np.delete(cam,np.where(cam[2,:]<0)[1],axis=1)
cam[:2] /= cam[2,:]
print(cam)
print(cam.shape)

plt.figure()
png = mpimg.imread(img)
IMG_H,IMG_W,_ = png.shape
plt.axis([0,IMG_W,IMG_H,0])
plt.imshow(png)

u,v,z = cam
# print(cam.shape)
# print(cam)
u_out = np.logical_or(u<0, u>IMG_W)
v_out = np.logical_or(v<0, v>IMG_H)
outlier = np.logical_or(u_out, v_out)
cam = np.delete(cam,np.where(outlier),axis=1)

plt.scatter([u],[v],c=[z],cmap='rainbow_r',alpha=0.5,s=1)
plt.title('name')
plt.savefig(f'./lidar2cam.png',bbox_inches='tight')
plt.show()


