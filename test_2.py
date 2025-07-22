import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, cm
import matplotlib as mpl
import pandas as pd
import os
from datetime import datetime

from scipy import interpolate
from scipy.interpolate import interp2d, RectBivariateSpline

# 设置目录和文件
direct = ['D:/2024cwa/no_use/test_folder/']
file = os.listdir('D:/2024cwa/no_use/test_folder/')

# 设置颜色映射
def setcolorQPF(cnum=1, initgray=True):
    if cnum == 1:
        clevs = [0.5, 1, 2, 5, 10, 15, 20, 30, 40, 50, 70, 90, 110, 130, 150, 200, 300]
    if cnum == 2:
        clevs = [0, 0.5, 2, 5, 10, 15, 20, 30, 40, 50, 70, 90, 110, 130, 150, 200, 300]
    if cnum == 3:
        clevs = [0, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 10, 15, 20, 25, 30, 40, 50, 80]
    if cnum == 4:
        clevs = [0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2, 3, 4, 5, 6, 7, 8]

    initrgb = (194, 194, 194) if initgray else (255, 255, 255)
    cmaplist = np.array([
        initrgb,
        (156, 252, 255),
        (3, 200, 255),
        (5, 155, 255),
        (3, 99, 255),
        (5, 153, 2),
        (57, 255, 3),
        (255, 251, 3),
        (255, 200, 0),
        (255, 149, 0),
        (255, 0, 0),
        (204, 0, 0),
        (153, 0, 0),
        (150, 0, 153),
        (201, 0, 204),
        (251, 0, 255)])

    return clevs, cmaplist

# 获取颜色等级和颜色映射
clevs, cmaplist = setcolorQPF(cnum=2)
cmap = (mpl.colors.ListedColormap(cmaplist/255.).with_extremes(
    over=(253/255., 201/255., 255/255.), under='white'))
norm = mpl.colors.BoundaryNorm(clevs, cmap.N)
clevs_labels = [str(v) if isinstance(v, float) else str(int(v)) for v in clevs]

# 解析文件名中的日期
def get_date_from_filename(filename):
    date_str = filename.split('_')[-1].split('.')[0]
    return datetime.strptime(date_str, "%Y%m%d%H%M")

# 将文件按日期分组
file_dates = [get_date_from_filename(f) for f in file]
date_groups = {}
for idx, date in enumerate(file_dates):
    date_str = date.strftime("%Y-%m-%d")
    if date_str not in date_groups:
        date_groups[date_str] = []
    date_groups[date_str].append(file[idx])

# 处理每一天的数据
for date_str, files in date_groups.items():
    formatted_date = date.strftime('%Y-%m-%d')
    hourly_qpe = np.zeros((24, 248, 200))
    m = Basemap(projection='cyl', resolution='i', fix_aspect=True, llcrnrlon=118, llcrnrlat=19.99, urcrnrlon=123.5, urcrnrlat=26.99, lat_ts=20)
    for b in range(len(files)):
        print(b)
        with open(direct[0] + files[b], "rb") as fid:
            test = np.fromfile(fid, dtype='float32')
            test = np.reshape(test, (248, 200))
            test = test[::-1, :]
            test[test <= 0] = np.nan

            theLons = np.arange(119, 122.98, 0.02)
            theLats = np.arange(21.02, 25.96, 0.02)

            filename = files[b]
            date_str = filename.split('_')[-1]
            tt11 = date_str.split('.')[0]

            hour = int(tt11[8:10])
            if hour < 24:
                hourly_qpe[hour] = test
            else:
                print('not used')

    daily_qpe = np.nansum(hourly_qpe, axis=0)

    # 绘制每日累计降雨量的地图
    plt.figure(dpi=600)
    m = Basemap(projection='cyl', resolution='i', fix_aspect=True, llcrnrlon=118, llcrnrlat=19.99, urcrnrlon=123.5, urcrnrlat=26.99, lat_ts=20)
    cx, cy = np.float32(np.meshgrid(theLons, theLats))
    c = m.contourf(cx, cy, daily_qpe, clevs, shading='gouraud', cmap=cmap, norm=norm)

    y_axis = m.drawparallels(np.arange(21, 26, 2), labels=[1, 0, 0, 0], color='#787878', fontsize=10)
    x_axis = m.drawmeridians(np.arange(119, 123.01, 2), labels=[0, 1, 0, 1], color='#787878', fontsize=10)
    m.drawcoastlines(color='#28FF28')
    plt.colorbar(c).set_label('Rainfall rate(mm/hr)')
    formatted_date = date.strftime('%Y-%m-%d')
    plt.title(f"Daily QPE for {formatted_date}", fontsize=12)

    plt.savefig(f"daily_qpe_{date_str}.png")
    plt.show()