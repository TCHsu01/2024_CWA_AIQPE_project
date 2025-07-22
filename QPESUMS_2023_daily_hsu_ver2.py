#%%
import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.basemap import Basemap, cm
import matplotlib as mpl
import pprint
import pandas as pd
#import cartopy.crs as ccrs
#from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
from haversine import haversine
import math
import os
#from geopy import distance
import seaborn as sns
import matplotlib.colors as mcolors
from scipy import interpolate
import h5py
import xarray as xr
from scipy.interpolate import interp2d
from datetime import datetime, timedelta

# 定義colorbar 顏色
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


direct=['H:/2024cwa/QPESUMS_2024實習/']
file=os.listdir('H:/2024cwa/QPESUMS_2024實習/')

#%%
# # 設定原始資料長度資訊
# days = [365,182]
# m = Basemap(projection='cyl', resolution='i',llcrnrlon=118, llcrnrlat=20, urcrnrlon=123.4, urcrnrlat=26.9)
# all_combination = []
# theLons = np.arange(118,123.49,0.1)
# theLats = np.arange(20,26.99,0.1)
# cx, cy = np.float32(np.meshgrid(theLons, theLats))
# # 透過長迴圈讀取2023、2024資料，並輸出單一.npy 檔案儲存全資料array
# for a in range(len(direct)):
#     combined_test = []
#     for b in range(len(days)):
#         with open(direct[a]+file[b], "rb") as fid :
#             test = np.fromfile(fid,np.float32)
#             test = np.asarray(test,dtype = float)
#             test[test<0] = np.nan
#             test = np.reshape(test,(days[b],70,55))
#             test = np.asarray(test,dtype = float)
#             for i in range(days[b]):
#                 for j in range(70):
#                     for k in range(55):
#                         # 海洋還是陸地的範圍判斷式
#                         # 海陸要反過來就改這裡就好
#                         lon, lat = m(cx[j, k], cy[j, k])
#                         is_land = m.is_land(lon, lat)
#                         if not is_land:
#                             test[i, j, k] = np.nan
#             # combined_test在這個迴圈是在儲存單一年的資料
#             combined_test.append(test)
#     combined_test = np.concatenate(combined_test, axis=0)
#       # 將年際資料合併在一起
#     all_combination.append(combined_test)
# # 因為array是四維的，squeeze將多餘的維度壓縮
# all_combination = np.array(all_combination)
# all_combination = np.squeeze(all_combination[:,:,20:55,20:45])
# # 輸出最終資料
# # np.save(f'QPESUMS_daily_all.npy',all_combination)
#%%
# 將 .npy讀入
all_combination = np.load('QPESUMS_daily_all.npy')

# 生成對應data的時間序列
base_time = datetime.strptime('2023-01-01 00:00', '%Y-%m-%d %H:%M')
step = timedelta(days=1)
time_full = [base_time + step * i for i in range(all_combination.shape[0])]

#%%
clevs, cmaplist = setcolorQPF(cnum=2)
cmap = (mpl.colors.ListedColormap(cmaplist/255.).with_extremes(
        over=(253/255., 201/255., 255/255.), under='white'))
norm = mpl.colors.BoundaryNorm(clevs, cmap.N)
clevs_labels = [str(v) if isinstance(v, float) else str(int(v)) for v in clevs]
theLons = np.arange(118,123.49,0.1)
theLats = np.arange(20,26.99,0.1)
cx, cy = np.float32(np.meshgrid(theLons, theLats))
cx = cx[20:55,20:45]
cy = cy[20:55,20:45]

# 時間範圍
start_date = datetime.strptime('2023-05-01', '%Y-%m-%d')
end_date = datetime.strptime('2023-05-03', '%Y-%m-%d')
indices = [i for i, t in enumerate(time_full) if start_date <= t <= end_date]

# 設定一個array儲存時間範圍內的平均降水
tar_QPESUMS = []

for tar in indices:
    tar_QPESUMS.append(all_combination[tar])
    m = Basemap(projection='cyl', resolution='i', fix_aspect=True, llcrnrlon=120, llcrnrlat=21.5, urcrnrlon=122, urcrnrlat=25.5, lat_ts=20)
    plt.figure(dpi=600)
    m.drawcoastlines(color='#28FF28') 
    CS = m.contourf(cx,cy,all_combination[tar],clevs,cmap=cmap,norm=norm)

    y_axis = m.drawparallels(np.arange(21.5, 25.51, 2), labels=[1, 0, 0, 0], color='#787878', fontsize=10)
    x_axis = m.drawmeridians(np.arange(120, 122.01, 2), labels=[0, 1, 0, 1], color='#787878', fontsize=10)
    plt.colorbar(CS).set_label('Rainfall rate(mm/day)')
    plt.title(f'QPESUMS daily rainfall in {time_full[tar]}',fontsize=12)
    date_str = time_full[tar].strftime("%Y%m%d")
    # filename_png = f"QPESUMS_daily_{date_str}.png"
    # plt.savefig(f"QPESUMS_daily_{date_str}.png")
    np.save(f'QPESUMS_daily_{date_str}.npy',all_combination[tar])
    plt.show()
#%%
# np.save('QPESUMS_daily_full_2023_usedin1103.npy',test)
# 取mean_QPESUMS作為處理資料範圍的平均
mean_QPESUMS = np.nanmean(tar_QPESUMS , axis=(0))
cx, cy = np.float32(np.meshgrid(theLons, theLats))
cx = cx[20:55,20:45]
cy = cy[20:55,20:45]
m = Basemap(projection='cyl', resolution='i' , fix_aspect=True,llcrnrlon=118, llcrnrlat=20, urcrnrlon=123.5 , urcrnrlat=27.00,lat_ts =20)
plt.figure(dpi=600)
m.drawcoastlines(color='#28FF28') 
clevs, cmaplist = setcolorQPF(cnum=2)
cmap = (mpl.colors.ListedColormap(cmaplist/255.).with_extremes(
        over=(253/255., 201/255., 255/255.), under='white'))
norm = mpl.colors.BoundaryNorm(clevs, cmap.N)
clevs_labels = [str(v) if isinstance(v, float) else str(int(v)) for v in clevs]
CS = m.contourf(cx,cy,mean_QPESUMS,clevs,cmap=cmap,norm=norm)
y_axis = m.drawparallels(np.arange(20,27.001,2),labels=[1,0,0,0], color='#787878')
x_axis = m.drawmeridians(np.arange(118,123.501,2),labels=[0,1,0,1], color='#787878')
plt.colorbar(CS).set_label('Rainfall rate(mm/day)')

d_1 = start_date.strftime("%Y-%m-%d");d_end = end_date.strftime("%Y-%m-%d")
plt.title(f'QPESUMS mean daily rainfall from {d_1} to {d_end}')

# 這裡還可以設定時間範圍內的.npy輸出，看有沒有需要而已
# fn_1 = start_date.strftime("%Y%m%d");fn_end = end_date.strftime("%Y%m%d")
# np.save(f'QPESUMS_daily_{fn_1}_{fn_end}.npy',mean_QPESUMS)

# 不知道有沒有需要accumulate rainfall，但其實就把上面的nanmean改成sum就可以了
