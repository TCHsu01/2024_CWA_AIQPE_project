import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, cm
import matplotlib as mpl
import pandas as pd
import os
import datetime
from datetime import date
from datetime import datetime

from scipy import interpolate
from scipy.interpolate import interp2d, RectBivariateSpline

# 设置目录和文件
direct = ['D:/2024cwa/no_use/test_folder/']
file = os.listdir('D:/2024cwa/no_use/test_folder/')

# 设置颜色映射
def setcolorQPF(cnum=1, initgray=True):
    if cnum==1: clevs = [0.5, 1, 2, 5, 10, 15, 20, 30, 40, 50, 70, 90, 110, 130, 150, 200, 300]
    if cnum==2: clevs = [0, 0.5, 2, 5, 10, 15, 20, 30, 40, 50, 70, 90, 110, 130, 150, 200, 300]
    if cnum==3: clevs = [0, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 10, 15, 20, 25, 30, 40, 50, 80]
    if cnum==4: clevs = [0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2, 3, 4, 5, 6, 7, 8]
    
    initrgb = (194, 194, 194) if initgray else (255, 255, 255)
    cmaplist = np.array([
            initrgb,
            (156, 252, 255),
            (  3, 200, 255),
            (  5, 155, 255),
            (  3,  99, 255),
            (  5, 153,   2),
            ( 57, 255,   3),
            (255, 251,   3),
            (255, 200,   0),
            (255, 149,   0),
            (255,   0,   0),
            (204,   0,   0),
            (153,   0,   0),
            (150,   0, 153),
            (201,   0, 204),
            (251,   0, 255)])
    
    return clevs, cmaplist

# 获取颜色等级和颜色映射
clevs, cmaplist = setcolorQPF(cnum=2)
cmap = (mpl.colors.ListedColormap(cmaplist/255.).with_extremes(
        over=(253/255., 201/255., 255/255.), under='white'))
norm = mpl.colors.BoundaryNorm(clevs, cmap.N)
clevs_labels = [str(v) if isinstance(v, float) else str(int(v)) for v in clevs]

# 定义时间范围和目标文件名
date_range = pd.date_range(start='2024-05-09 00:00', end='2024-05-10 23:00', freq='H')
target_file = [f"model_hourly_{dt.strftime('%Y%m%d%H%M')}.bin" for dt in date_range]

# 将列表每 24 项分为一组的函数
def split_list(lst, n):
    return [lst[i:i + n] for i in range(0, len(lst), n)]

# 使用函数将列表分组
daily_grouped_fn = split_list(target_file, 24)

# 初始化数据数组
hourly_qpe = np.zeros((25, 248, 200))
hour_remind = np.zeros((25))
day_remind = np.zeros((25))
m = Basemap(projection='cyl', resolution='i', fix_aspect=True, llcrnrlon=118, llcrnrlat=19.99, urcrnrlon=123.5, urcrnrlat=26.99, lat_ts=20)

# 遍历目录和文件
for a in range(len(direct)):
    for b in range(len(target_file)):  # replace  len(file)
        print(b)
        with open(direct[a]+file[b], "rb") as fid:
            test = np.fromfile(fid, dtype='float32')
            test = np.reshape(test, (248, 200))
            test = test[::-1, :]
            test[test <= 0] = np.nan

            theLons = np.arange(119, 122.98, 0.02)
            theLats = np.arange(21.02, 25.96, 0.02)
            
            filename = file[b]
            date_str = filename.split('_')[-1]  # Extract the part before the last dot
            tt11 = date_str.split('.')[0]

            #######################################################################################
            if int(tt11[8:10]) < 24:  # Check if the hour is valid (0 to 23)
                datetime_var = datetime.strptime(tt11, "%Y%m%d%H%M")
                datetime_var = datetime_var.strftime("%Y-%m-%d %H:%M")

                hourly_qpe[int(tt11[8:10])] = test
                hour_remind[int(tt11[8:10])] = int(tt11[8:10])
                day_remind[int(tt11[8:10])] = int(tt11[6:8])
            else:
                print('not used')
            ##########################################################################################

#########################################################################################            

# 计算每日降雨量
num = np.round(len(hourly_qpe) / 24, decimals=0)
daily_qpe = np.zeros((int(num), 248, 200))

for dd in range(len(daily_qpe)):
    if (int(hour_remind[dd+1]) - int(hour_remind[dd]) == 1) or (int(hour_remind[dd+1]) - int(hour_remind[dd]) == -23):
        print('good')
    else:
        print('data missing or incorrect')

for dd in range(len(daily_qpe)):          
    daily_qpe[dd] = np.nansum(hourly_qpe[dd*24:(dd+1)*24], axis=0)
    
# 绘制地图
plt.figure(dpi=600)
m = Basemap(projection='cyl', resolution='i', fix_aspect=True, llcrnrlon=118, llcrnrlat=19.99, urcrnrlon=123.5, urcrnrlat=26.99, lat_ts=20)
cx, cy = np.float32(np.meshgrid(theLons, theLats))
c = m.contourf(cx, cy, daily_qpe[0], clevs, shading='gouraud', cmap=cmap, norm=norm)

y_axis = m.drawparallels(np.arange(21, 26, 2), labels=[1, 0, 0, 0], color='#787878', fontsize=10)
x_axis = m.drawmeridians(np.arange(119, 123.01, 2), labels=[0, 1, 0, 1], color='#787878', fontsize=10)
m.drawcoastlines(color='#28FF28')
plt.colorbar(c).set_label('Rainfall rate(mm/hr)')

plt.title(f"Daily QPE for {datetime_var[0:10]}", fontsize=12)
plt.show()
