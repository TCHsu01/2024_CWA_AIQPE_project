#%%
import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.basemap import Basemap, cm
import matplotlib as mpl
import re

import pprint
import pandas as pd
#import cartopy.crs as ccrs
#from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
from haversine import haversine
import h5py
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
from scipy.stats import gaussian_kde
from matplotlib.ticker import FuncFormatter



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

# set path
direct = 'H:/2024cwa/comparison/'
file_list = os.listdir(direct)

QPESUMS_23 = np.load(os.path.join(direct, 'new_QPESUMS_daily_20230501_20230831.npy'))
QPESUMS_24 = np.load(os.path.join(direct, 'new_QPESUMS_daily_20240501_20240630.npy'))
QPESUMS_GM = np.load(os.path.join(direct, 'new_QPESUMS_daily_20240721_20240728.npy'))
QPESUMS_cb = np.concatenate((QPESUMS_23, QPESUMS_24, QPESUMS_GM), axis=0)

AI313_23 = np.load(os.path.join(direct, 'island_313_degrade_aiqpe_20230501_20230831.npy'))
AI313_24 = np.load(os.path.join(direct, 'island_313_degrade_aiqpe_20240501_20240630.npy'))
AI313_GM = np.load(os.path.join(direct, 'island_313_degrade_aiqpe_20240721_20240728.npy'))
AI313_cb = np.concatenate((AI313_23, AI313_24, AI313_GM), axis=0)

AI322_23 = np.load(os.path.join(direct, 'island_322_degrade_aiqpe_20230501_20230831.npy'))
AI322_24 = np.load(os.path.join(direct, 'island_322_degrade_aiqpe_20240501_20240630.npy'))
AI322_GM = np.load(os.path.join(direct, 'island_322_degrade_aiqpe_20240721_20240728.npy'))
AI322_cb = np.concatenate((AI322_23, AI322_24, AI322_GM), axis=0)

date_23 = np.load(os.path.join(direct, 'date_list_20230501_20230831.npy'))
date_24 = np.load(os.path.join(direct, 'date_list_20240501_20240630.npy'))
date_GM = np.load(os.path.join(direct, 'date_list_20240721_20240728.npy'))
date_cb = np.concatenate((date_23, date_24, date_GM), axis=0)

IMERGE = np.load(os.path.join(direct, 'island_IMERGE_230501_240601.npy'))

QPESUMS = QPESUMS_cb
AI313 = AI313_cb
AI322 = AI322_cb
date = date_cb
############################

def get_indices(date_cb, case_type, start_dates, end_dates):
    """
    获取指定日期范围内的索引。

    参数：
    date_cb (list): 日期字符串列表。
    case_type (str): 案例类型，可以是 'general', 'mayu', 'TC'。
    start_dates (list): 开始日期列表，格式为 'YYYY-MM-DD'。
    end_dates (list): 结束日期列表，格式为 'YYYY-MM-DD'。

    返回：
    indices (np.array): 满足条件的索引数组。
    """
    x = np.array([datetime.strptime(date_str, '%Y-%m-%d') for date_str in date_cb])
    indices = []
    
    for start_date, end_date in zip(start_dates, end_dates):
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
        indices_case = [i for i, t in enumerate(x) if start_date <= t <= end_date]
        indices.append(indices_case)
    
    # 将多个索引列表合并为一个
    indices = np.concatenate(indices)
    return indices
    # # 示例使用
    # date_cb = ['2023-04-01', '2023-05-15', '2023-06-20', '2024-05-10', '2024-06-25', '2024-07-30']
    # case_type = 'TC'
    # start_dates = ['2023-05-01', '2024-05-01']
    # end_dates = ['2023-06-30', '2024-06-30']

############################
# # # # # for general continuity case
start_dates = ['2023-05-01']
end_dates = [ '2024-06-01']
case_type = ' '
indices = get_indices(date_cb, case_type, start_dates, end_dates)
########################################################################
# # # # # # # for mayu case
# start_dates = ['2023-05-01', '2024-05-01']
# end_dates = ['2023-06-30', '2024-05-01']
# case_type = 'meiyu'
# indices = get_indices(date_cb, case_type, start_dates, end_dates)
########################################################################
# # # # # # # for TC case"s"
# start_dates = ['2023-05-29', '2023-07-24','2023-08-01','2023-08-28','2024-07-22']
# end_dates = ['2023-05-31', '2023-07-28','2023-08-04','2023-08-31','2024-07-26']
# start_dates = ['2024-07-22']
# end_dates = ['2024-07-26']

# case_type = 'typhoon'
# indices = get_indices(date_cb, case_type, start_dates, end_dates)
########################################################################
data = pd.DataFrame({'AI313 ': AI313[indices].flatten(),
                     'QPESUMS': QPESUMS[indices].flatten()})
# 计算相关系数
coef = data.corr()
coef = np.asarray(coef)
daily_corr1 = coef[0, 1]
bias1 = AI313 - QPESUMS
bias1 = np.nanmean(bias1)
print('AI313_CC = '+str(daily_corr1))
print('AI313_bias = '+str(bias1))

###################################################
data = pd.DataFrame({'AI322  ': AI322[indices].flatten(),
                     'QPESUMS': QPESUMS[indices].flatten()})
# 计算相关系数
coef = data.corr()
coef = np.asarray(coef)
daily_corr2 = coef[0, 1]
bias2 = AI322 - QPESUMS
bias2 = np.nanmean(bias2)
print('AI322_CC = '+str(daily_corr2))
print('AI322_bias = '+str(bias2))
###################################################
data = pd.DataFrame({'AI313  ': AI313[indices].flatten(),
                     'IMERGE': IMERGE[indices].flatten()})
# 计算相关系数
coef = data.corr()
coef = np.asarray(coef)
daily_corr = coef[0, 1]
bias = AI313[0:155,:,:] - IMERGE
bias = np.nanmean(bias)
print('AI313_IMERGE_CC = '+str(daily_corr))
print('AI313_IMERGE_bias = '+str(bias))

theLons = np.arange(120, 122.99, 0.1)##
theLats = np.arange(22, 25.99, 0.1)##
#%%
# 設定顏色等高線
clevs, cmaplist = setcolorQPF(cnum=2)
cmap = (mpl.colors.ListedColormap(cmaplist/255.).with_extremes(
        over=(253/255., 201/255., 255/255.), under='white'))
norm = mpl.colors.BoundaryNorm(clevs, cmap.N)
clevs_labels = [str(v) if isinstance(v, float) else str(int(v)) for v in clevs]

# 初始化數據列表
along_time_coeff_AI313 = []
along_time_bias_AI313 = []
mean_aipqe_AI313 = []
along_time_coeff_AI322 = []
along_time_bias_AI322 = []
along_time_coeff_IMERGE = []
along_time_bias_IMERGE = []
mean_aipqe_AI322 = []
mean_qpesums = []
mean_IMERGE = []
space_mean_qpesums = []
space_mean_ai313 = []
space_mean_ai322 = []
space_mean_IMERGE = []

for i in range(len(indices)):
    formatted_date = date[indices[i]]  # 日期
    #######
    data = pd.DataFrame({'AI313 ': AI313[indices[i]].flatten(), 'QPESUMS': QPESUMS[indices[i]].flatten()})
    coef = data.corr()
    coef = np.asarray(coef)
    time_daily_corr1 = coef[0, 1]
    time_bias1 = AI313[indices[i]] - QPESUMS[indices[i]]
    time_bias1 = np.nanmean(time_bias1)
    sumqpe1 = np.nanmean(AI313[indices[i]])
    along_time_coeff_AI313.append(time_daily_corr1)
    along_time_bias_AI313.append(time_bias1)
    mean_aipqe_AI313.append(sumqpe1)
    #######
    data = pd.DataFrame({'AI322 ': AI322[indices[i]].flatten(), 'QPESUMS': QPESUMS[indices[i]].flatten()})
    coef = data.corr()
    coef = np.asarray(coef)
    time_daily_corr2 = coef[0, 1]
    time_bias2 = AI322[indices[i]] - QPESUMS[indices[i]]
    time_bias2 = np.nanmean(time_bias2)
    sumqpe2 = np.nanmean(AI322[indices[i]])
    along_time_coeff_AI322.append(time_daily_corr2)
    along_time_bias_AI322.append(time_bias2)
    mean_aipqe_AI322.append(sumqpe2)
    #######
    data = pd.DataFrame({'AI313 ': AI313[indices[i]].flatten(), 'IMERGE': IMERGE[indices[i]].flatten()})
    coef = data.corr()
    coef = np.asarray(coef)
    time_daily_corr3 = coef[0, 1]
    time_bias3 = AI313[indices[i]] - IMERGE[indices[i]]
    time_bias3 = np.nanmean(time_bias3)
    sumqpe3 = np.nanmean(IMERGE[indices[i]])
    along_time_coeff_IMERGE.append(time_daily_corr3)
    along_time_bias_IMERGE.append(time_bias3)
    mean_IMERGE.append(sumqpe3)

    sumqpesum = np.nanmean(QPESUMS[indices[i]])
    mean_qpesums.append(sumqpesum)
    ######
    # # 出圖的部分
    # fig, axs = plt.subplots(2, 2, figsize=(15, 5), constrained_layout=True, facecolor='white')
    # m = Basemap(projection='cyl', resolution='i', fix_aspect=True,
    #     llcrnrlon=120, llcrnrlat=22, urcrnrlon=122.5, urcrnrlat=25.5, lat_ts=22)

    # # 第一張：QPESUM
    # ax = axs[0]
    # cx, cy = np.meshgrid(theLons, theLats)
    # c = m.contourf(cx, cy, QPESUMS[indices[i]], clevs, cmap=cmap, norm=norm, ax=ax)
    # m.drawparallels(np.arange(22, 25.5, 2), labels=[1, 0, 0, 0], color='#787878', fontsize=12, ax=ax)
    # m.drawmeridians(np.arange(120, 122.5, 2), labels=[0, 1, 0, 1], color='#787878', fontsize=12, ax=ax)
    # m.drawcoastlines(color='#28FF28', ax=ax)
    
    # ax.set_title(f"QPESUM for {date[indices[i]]}", fontsize=12)

    # # 第二張：AI 313
    # ax = axs[1]
    # cx, cy = np.meshgrid(theLons, theLats)
    # c = m.contourf(cx, cy, AI313[indices[i]], clevs, cmap=cmap, norm=norm, ax=ax)
    # m.drawparallels(np.arange(22, 25.5, 2), labels=[1, 0, 0, 0], color='#787878', fontsize=12, ax=ax)
    # m.drawmeridians(np.arange(120, 122.5, 2), labels=[0, 1, 0, 1], color='#787878', fontsize=12, ax=ax)
    # m.drawcoastlines(color='#28FF28', ax=ax)
    # axs[1].annotate(f'CC: {time_daily_corr1:.2f}\nBias: {time_bias1:.2f}', xy=(0.97, 0.97), 
    #             xycoords='axes fraction', horizontalalignment='right', verticalalignment='top',
    #             bbox=dict(facecolor='white', alpha=0.8))
    # ax.set_title(f"AI 313 for {date[indices[i]]}", fontsize=12)

    # # 第三張：AI 322
    # ax = axs[2]
    # c = m.contourf(cx, cy, AI322[indices[i]], clevs, cmap=cmap, norm=norm, ax=ax)
    # m.drawparallels(np.arange(22, 25.5, 2), labels=[1, 0, 0, 0], color='#787878', fontsize=12, ax=ax)
    # m.drawmeridians(np.arange(120, 122.5, 2), labels=[0, 1, 0, 1], color='#787878', fontsize=12, ax=ax)
    # m.drawcoastlines(color='#28FF28', ax=ax)
    # axs[2].annotate(f'CC: {time_daily_corr2:.2f}\nBias: {time_bias2:.2f}', xy=(0.97, 0.97), 
    #             xycoords='axes fraction', horizontalalignment='right', verticalalignment='top',
    #             bbox=dict(facecolor='white', alpha=0.8))
    # ax.set_title(f"AI 322 for {date[indices[i]]}", fontsize=12)
    # # 第四張：IMERGE
    # ax = axs[3]
    # c = m.contourf(cx, cy, IMERGE[indices[i]], clevs, cmap=cmap, norm=norm, ax=ax)
    # m.drawparallels(np.arange(22, 25.5, 2), labels=[1, 0, 0, 0], color='#787878', fontsize=12, ax=ax)
    # m.drawmeridians(np.arange(120, 122.5, 2), labels=[0, 1, 0, 1], color='#787878', fontsize=12, ax=ax)
    # m.drawcoastlines(color='#28FF28', ax=ax)
    # axs[3].annotate(f'CC: {time_daily_corr3:.2f}\nBias: {time_bias3:.2f}', xy=(0.97, 0.97), 
    #             xycoords='axes fraction', horizontalalignment='right', verticalalignment='top',
    #             bbox=dict(facecolor='white', alpha=0.8))
    # ax.set_title(f"IMERGE for {date[indices[i]]}", fontsize=12)
    # # 共用 colorbar
    # fig.colorbar(c, ax=axs, orientation='vertical', fraction=.02, pad=0.04).set_label('Rainfall rate(mm/hr)')
    
    # valid_filename = re.sub(r'[^\w\-_\. ]', '_', f"island_comparison_{formatted_date}.png")
    # # plt.savefig(os.path.join(direct, valid_filename))  # 使用绝对路径
    # # plt.savefig(valid_filename)
    # plt.show()
    space_mean_qpesums.append(QPESUMS[indices[i]])
    space_mean_ai313.append(AI313[indices[i]])
    space_mean_ai322.append(AI322[indices[i]])
    space_mean_IMERGE.append(IMERGE[indices[i]])
########################################################
# # # # # # space mean # # # # # #
# 转换为 numpy 数组，方便后续计算
space_mean_qpesums = np.array(space_mean_qpesums)
space_mean_ai313 = np.array(space_mean_ai313)
space_mean_ai322 = np.array(space_mean_ai322)

# 计算平均值，确保在计算时维度是正确的
space_mean_qpesums = np.nanmean(space_mean_qpesums, axis=0)
space_mean_ai313 = np.nanmean(space_mean_ai313, axis=0)
space_mean_ai322 = np.nanmean(space_mean_ai322, axis=0)
space_mean_IMERGE = np.nanmean(space_mean_IMERGE, axis=0)

# space mean plot
fig, axs = plt.subplots(1, 4, figsize=(15, 5), constrained_layout=True, facecolor='white')
m = Basemap(projection='cyl', resolution='i', fix_aspect=True,
    llcrnrlon=120, llcrnrlat=22, urcrnrlon=122.5, urcrnrlat=25.5, lat_ts=22)

# 第一張：QPESUM
ax = axs[0]
cx, cy = np.meshgrid(theLons, theLats)
c = m.contourf(cx, cy,space_mean_qpesums, clevs, cmap=cmap, norm=norm, ax=ax)
m.drawparallels(np.arange(22, 25.5, 2), labels=[1, 0, 0, 0], color='#787878', fontsize=12, ax=ax)
m.drawmeridians(np.arange(120, 122.5, 2), labels=[0, 1, 0, 1], color='#787878', fontsize=12, ax=ax)
m.drawcoastlines(color='#28FF28', ax=ax)

ax.set_title(f"QPESUM from {date[indices[0]]} to {date[indices[-1]]}", fontsize=12)

# 第二張：AI 313
ax = axs[1]
cx, cy = np.meshgrid(theLons, theLats)
c = m.contourf(cx, cy,space_mean_ai313, clevs, cmap=cmap, norm=norm, ax=ax)
m.drawparallels(np.arange(22, 25.5, 2), labels=[1, 0, 0, 0], color='#787878', fontsize=12, ax=ax)
m.drawmeridians(np.arange(120, 122.5, 2), labels=[0, 1, 0, 1], color='#787878', fontsize=12, ax=ax)
m.drawcoastlines(color='#28FF28', ax=ax)
axs[1].annotate(f'CC: {np.nanmean(along_time_coeff_AI313):.2f}\nBias: {np.nanmean(along_time_bias_AI313):.2f}', xy=(0.97, 0.97), 
            xycoords='axes fraction', horizontalalignment='right', verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.8))
ax.set_title(f"AI 313 from {date[indices[0]]} to {date[indices[-1]]}", fontsize=12)

# 第三張：AI 322
ax = axs[2]
c = m.contourf(cx, cy,space_mean_ai322, clevs, cmap=cmap, norm=norm, ax=ax)
m.drawparallels(np.arange(22, 25.5, 2), labels=[1, 0, 0, 0], color='#787878', fontsize=12, ax=ax)
m.drawmeridians(np.arange(120, 122.5, 2), labels=[0, 1, 0, 1], color='#787878', fontsize=12, ax=ax)
m.drawcoastlines(color='#28FF28', ax=ax)
axs[2].annotate(f'CC: {np.nanmean(along_time_coeff_AI322):.2f}\nBias: {np.nanmean(along_time_bias_AI322):.2f}', xy=(0.97, 0.97), 
            xycoords='axes fraction', horizontalalignment='right', verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.8))
ax.set_title(f"AI 322 from {date[indices[0]]} to {date[indices[-1]]}", fontsize=12)

# 第四張：IMERGE
ax = axs[3]
c = m.contourf(cx, cy,space_mean_IMERGE, clevs, cmap=cmap, norm=norm, ax=ax)
m.drawparallels(np.arange(22, 25.5, 2), labels=[1, 0, 0, 0], color='#787878', fontsize=12, ax=ax)
m.drawmeridians(np.arange(120, 122.5, 2), labels=[0, 1, 0, 1], color='#787878', fontsize=12, ax=ax)
m.drawcoastlines(color='#28FF28', ax=ax)
axs[3].annotate(f'CC: {np.nanmean(along_time_coeff_IMERGE):.2f}\nBias: {np.nanmean(along_time_bias_IMERGE):.2f}', xy=(0.97, 0.97), 
            xycoords='axes fraction', horizontalalignment='right', verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.8))
ax.set_title(f"IMERGE from {date[indices[0]]} to {date[indices[-1]]}", fontsize=12)


# 共用 colorbar
fig.colorbar(c, ax=axs, orientation='vertical', fraction=.02, pad=0.04).set_label('Rainfall rate(mm/hr)')

# valid_filename = re.sub(r'[^\w\-_\. ]', '_', f"island_comparison_{formatted_date}.png")
# plt.savefig(os.path.join(direct, "GAEMI_240723_240726_space_mean.png"))  # 使用绝对路径
# # plt.savefig(valid_filename)
# plt.show()
#%%
fig, ax = plt.subplots(dpi=600, facecolor='white')
midium_coeff_313 = np.nanmedian(along_time_coeff_AI313, axis=None, out=None, overwrite_input=False, keepdims=False)
midium_coeff_322 = np.nanmedian(along_time_coeff_AI322, axis=None, out=None, overwrite_input=False, keepdims=False)
axx = np.arange(1,193)
plt.scatter(mean_qpesums,mean_aipqe_AI313,alpha=0.5,label='AIQPE313')
plt.scatter(mean_qpesums,mean_aipqe_AI322,alpha=0.5,label='AIQPE322')
plt.xlabel('QPESUMS')
plt.ylabel('AI IRQPE')
plt.legend()
x = np.arange(0,1000,1)
plt.plot(x,x);plt.xlim(0,100);plt.ylim(0,100)
# plt.savefig(os.path.join(direct,"all_mean_coefficient_plot.png"));plt.show()
#%%
# 假设你已经定义好了 x
x = np.linspace(0.5, 1000, 1000)  # 避免 log10(0) 错误

flat_QPESUMS_cb = QPESUMS[indices].flatten() 
flat_AI313_cb = AI313[indices].flatten()
flat_AI322_cb = AI322[indices].flatten()
# # # # # # all data # # # # # #
# flat_QPESUMS_cb = QPESUMS_cb.flatten() 
# flat_AI313_cb = AI313_cb.flatten()
# flat_AI322_cb = AI322_cb.flatten()

data_flat = pd.DataFrame({'AI 322': flat_AI322_cb,
                          'QPESUMS': flat_QPESUMS_cb,
                          'AI 313': flat_AI313_cb})
data_flat_clean = data_flat.replace([np.inf, -np.inf], np.nan).dropna()

# 计算相关系数
correlation_coefficient313 = np.corrcoef(data_flat_clean['QPESUMS'], data_flat_clean['AI 313'])[0, 1]
correlation_coefficient322 = np.corrcoef(data_flat_clean['QPESUMS'], data_flat_clean['AI 322'])[0, 1]
# 计算 RMSE
diff = np.subtract(data_flat_clean['QPESUMS'], data_flat_clean['AI 313'])
square = np.square(diff);MSE = square.mean();RMSE313 = np.sqrt(MSE)
diff = np.subtract(data_flat_clean['QPESUMS'], data_flat_clean['AI 322'])
square = np.square(diff);MSE = square.mean();RMSE322 = np.sqrt(MSE)
# 计算高斯密度估计
xy_all = np.vstack([data_flat_clean['QPESUMS'], data_flat_clean['AI 313']])
z313 = gaussian_kde(xy_all)(xy_all)
xy_all = np.vstack([data_flat_clean['QPESUMS'], data_flat_clean['AI 322']])
z322 = gaussian_kde(xy_all)(xy_all)

# 移除包含NaN或inf的行
data_flat_clean = data_flat.replace([np.inf, -np.inf], np.nan).dropna()
#%%
from scipy.stats import pearsonr

# 初始化存储空间相关性的数组
spatial_corr_AI313 = np.zeros_like(IMERGE[0])
# spatial_corr_AI322 = np.zeros_like(IMERGE[0])
spatial_corr_IMERGE = np.zeros_like(IMERGE[0])

# 遍历每个网格点，计算空间相关性
for lat in range(IMERGE.shape[1]):
    for lon in range(IMERGE.shape[2]):
        # 提取时间序列
        # series_QPESUMS = QPESUMS[indices[0]:indices[-1], lat, lon]
        series_AI313 = AI313[indices[0]:indices[-1], lat, lon]
        # series_AI322 = AI322[indices[0]:indices[-1], lat, lon]
        series_IMERGE = IMERGE[indices[0]:indices[-1], lat, lon]
        
        # # 计算AI313与QPESUMS的相关系数
        # if not np.isnan(series_QPESUMS).all() and not np.isnan(series_AI313).all():
        #     spatial_corr_AI313[lat, lon], _ = pearsonr(series_QPESUMS, series_AI313)
        # else:
        #     spatial_corr_AI313[lat, lon] = np.nan

        # # 计算AI322与QPESUMS的相关系数
        # if not np.isnan(series_QPESUMS).all() and not np.isnan(series_AI322).all():
        #     spatial_corr_AI322[lat, lon], _ = pearsonr(series_QPESUMS, series_AI322)
        # else:
        #     spatial_corr_AI322[lat, lon] = np.nan

        # 计算AI313与IMERGE的相关系数
        if not np.isnan(series_IMERGE).all() and not np.isnan(series_AI313).all():
            spatial_corr_IMERGE[lat, lon], _ = pearsonr(series_IMERGE, series_AI313)
        else:
            spatial_corr_IMERGE[lat, lon] = np.nan

# 绘制相关系数图
fig, axs = plt.subplots(1, 1, figsize=(10, 5), constrained_layout=True, facecolor='white')
fig.subplots_adjust(wspace=0.05)  # 减小子图之间的宽度间距
m = Basemap(projection='cyl', resolution='i', fix_aspect=True,
            llcrnrlon=120, llcrnrlat=22, urcrnrlon=122.5, urcrnrlat=25.5, lat_ts=22)
cx, cy = np.meshgrid(theLons, theLats)

# 绘制AI313相关系数图
# ax = axs[0]
# cx, cy = np.meshgrid(theLons, theLats)
# c = m.contourf(cx, cy, spatial_corr_AI313, np.linspace(0, 1, 21), cmap='rainbow', norm=mpl.colors.Normalize(vmin=0, vmax=1), ax=ax)
# m.drawparallels(np.arange(22, 25.5, 2), labels=[1, 0, 0, 0], color='#787878', fontsize=12, ax=ax)
# m.drawmeridians(np.arange(120, 122.5, 2), labels=[0, 1, 0, 1], color='#787878', fontsize=12, ax=ax)
# m.drawcoastlines(color='#28FF28', ax=ax)
# ax.set_title(f"Spatial Correlation AI313 vs QPESUMS {case_type}", fontsize=12)

# # 绘制AI322相关系数图
# ax = axs[1]
# c = m.contourf(cx, cy, spatial_corr_AI322, np.linspace(0, 1, 21), cmap='rainbow', norm=mpl.colors.Normalize(vmin=0, vmax=1), ax=ax)
# m.drawparallels(np.arange(22, 25.5, 2), labels=[1, 0, 0, 0], color='#787878', fontsize=12, ax=ax)
# m.drawmeridians(np.arange(120, 122.5, 2), labels=[0, 1, 0, 1], color='#787878', fontsize=12, ax=ax)
# m.drawcoastlines(color='#28FF28', ax=ax)
# ax.set_title(f"Spatial Correlation AI322 vs QPESUMS {case_type}", fontsize=12)
ax = axs
c = m.contourf(cx, cy, spatial_corr_IMERGE, np.linspace(0, 1, 21), cmap='rainbow', norm=mpl.colors.Normalize(vmin=0, vmax=1), ax=ax)
m.drawparallels(np.arange(22, 25.5, 2), labels=[1, 0, 0, 0], color='#787878', fontsize=12, ax=ax)
m.drawmeridians(np.arange(120, 122.5, 2), labels=[0, 1, 0, 1], color='#787878', fontsize=12, ax=ax)
m.drawcoastlines(color='#28FF28', ax=ax)
ax.set_title(f"Spatial Correlation AI313 vs IMERGE {case_type}", fontsize=12)
# 共用 colorbar
fig.colorbar(c, ax=axs, orientation='vertical', fraction=.02, pad=0.04).set_label('Correlation Coefficient')

# 保存和显示图像
# valid_filename = re.sub(r'[^\w\-_\. ]', '_', f"spatial_correlation_{case_type}.png")
# plt.savefig(valid_filename)  # 使用绝对路径
plt.show()
#%%
from matplotlib.ticker import FuncFormatter, FixedLocator
# 自定义科学记数法刻度标签函数，使用上标形式
def scientific_formatter(x, pos):
    if x == 0:
        return "0"
    exponent = int(np.log10(x))
    return f'$10^{{{exponent}}}$'

# 画图函数
def plot_data(ax, x_data, y_data, density, xlabel, ylabel, title, RMSE, corr):
    sc = ax.scatter(x_data, y_data, c=density, s=1, cmap='turbo')
    ax.plot(x, x, color='red')
    ax.set_xlim(0.5, 1000);ax.set_ylim(0.5, 1000)
    # ax.set_xlim(0.001, 1000);ax.set_ylim(0.001, 1000)
    ax.set_xlabel(xlabel);ax.set_ylabel(ylabel);ax.set_title(title)
    ax.set_xscale('log');ax.set_yscale('log')
    # 0.001,0.01,0.1,1
    major_ticks = [10, 100, 1000]
    minor_ticks = [80, 200, 350, 500]

    ax.set_xticks(major_ticks);ax.set_xticks(minor_ticks, minor=True)
    ax.set_yticks(major_ticks);ax.set_yticks(minor_ticks, minor=True)

    ax.xaxis.set_major_formatter(FuncFormatter(scientific_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(scientific_formatter))

    ax.xaxis.set_minor_locator(FixedLocator(minor_ticks))
    ax.yaxis.set_minor_locator(FixedLocator(minor_ticks))

    ax.grid(which='both', color='gray', linestyle='--', linewidth=0.5, alpha=0.8)

    # 设置颜色限制
    cmax = 0.012
    sc.set_clim([0, cmax])
    
    cbar = plt.colorbar(sc, ax=ax, ticks=np.linspace(0, cmax, 7))
    cbar.set_label('Density')

    ax.annotate(f'RMSE: {RMSE:.2f}\ncorr: {corr:.2f}', 
                xy=(0.05, 0.95), 
                xycoords='axes fraction', 
                fontsize=12, 
                color='black',
                horizontalalignment='left', 
                verticalalignment='top',
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

# 更新 x 范围
x = np.linspace(0.5, 1000, 1000)
# x = np.linspace(0.001, 1000, 1000)

# 创建图像 313
fig, ax = plt.subplots(dpi=600, facecolor='white')
plot_data(ax, data_flat_clean['QPESUMS'], data_flat_clean['AI 313'], z313, 
          'QPESUMS', 'AI IRQPE 313', 'AI IRQPE 313 vs QPESUMS meiyu', RMSE313, correlation_coefficient313)
plt.savefig("AI_IRQPE_313_QPESUM_meiyu_flat_comparison.png")
plt.show()
# (Log Scale)
# _log_adjust_axis
# 创建图像 322
fig, ax = plt.subplots(dpi=600, facecolor='white')
plot_data(ax, data_flat_clean['QPESUMS'], data_flat_clean['AI 322'], z322, 
          'QPESUMS', 'AI IRQPE 322', 'AI IRQPE 322 vs QPESUMS meiyu', RMSE322, correlation_coefficient322)
plt.savefig("AI_IRQPE_322_QPESUM_meiyu_flat_comparison.png")
plt.show()
#%%
# violin
import seaborn as sns
import matplotlib.pyplot as plt
# 将两组数据组合成一个 DataFrame
data = pd.DataFrame({
    'Coefficient': np.concatenate([along_time_coeff_AI313, along_time_coeff_AI322]),
    'Group': ['AI 313'] * len(along_time_coeff_AI313) + ['AI 322'] * len(along_time_coeff_AI322)
})

# 设置主题
sns.set_theme(style="whitegrid")

# 设置 matplotlib figure
fig, ax = plt.subplots(figsize=(5, 6), dpi=900, facecolor='white')

# 绘制小提琴图
sns.violinplot(x='Group', y='Coefficient', data=data, bw_adjust=.5, cut=0, linewidth=1, palette="tab10",
inner_kws=dict(box_width=8, whis_width=1, color="0.3"))

# 设置 y 轴范围
ax.set(ylim=(-.7, 1.05))
ax.set_ylim(-0.2, 1)
ax.set_yticks(np.arange(-0.3, 1.1, 0.1))  # 从-0.2到1.0，每0.1一横
ax.set_yticklabels([f'{tick:.1f}' for tick in np.arange(-0.3, 1.1, 0.1)])  # 设置y轴刻度标签
ax.set_title(f"coefficient comparison {case_type}", fontsize=12)
# 去除边框
# sns.despine(left=True, bottom=True)
plt.savefig(f"violin_coefficient_{case_type}.png", bbox_inches='tight', pad_inches=0.1)
# _meiyu
# _typhoon
# 显示图像
plt.show()
#%%
# violin bias
import seaborn as sns
import matplotlib.pyplot as plt
# 将两组数据组合成一个 DataFrame
data = pd.DataFrame({
    'Bias': np.concatenate([along_time_bias_AI313, along_time_bias_AI322]),
    'Group': ['AI 313'] * len(along_time_bias_AI313) + ['AI 322'] * len(along_time_bias_AI322)
})

# 设置主题
sns.set_theme(style="whitegrid")

# 设置 matplotlib figure
fig, ax = plt.subplots(figsize=(5, 6), dpi=600, facecolor='white')

# 绘制小提琴图
sns.violinplot(x='Group', y='Bias', data=data, bw_adjust=.5, cut=0, linewidth=1, palette="tab10",
inner_kws=dict(box_width=8, whis_width=1, color="0.3"))

# 使用盒子图来增强四分位距条
# sns.boxplot(x='Group', y='Bias', data=data, whis=np.inf, linewidth=2.5, palette="Set3")

# 设置 y 轴范围
ax.set(ylim=(-120, 30))

ax.set_yticks(np.arange(-120, 30, 10))  # 从-150到30，每10一横
ax.set_yticklabels([f'{tick:.1f}' for tick in np.arange(-120, 30, 10)])  # 设置y轴刻度标签
ax.set_title(f"bias comparison {case_type}", fontsize=12)
# 去除边框
# sns.despine(left=True, bottom=True)
plt.savefig(f"violin_bias_{case_type}.png", bbox_inches='tight', pad_inches=0.1)
# _meiyu
# _typhoon
# 显示图像
plt.show()
#%%
# violin bias
import seaborn as sns
import matplotlib.pyplot as plt
# 将两组数据组合成一个 DataFrame
data = pd.DataFrame({
    'Bias': np.concatenate([along_time_bias_AI313, along_time_bias_IMERGE]),
    'Group': ['AI 313'] * len(along_time_bias_AI313) + ['IMERGE'] * len(along_time_bias_IMERGE)
})

# 设置主题
sns.set_theme(style="whitegrid")

# 设置 matplotlib figure
fig, ax = plt.subplots(figsize=(5, 6), dpi=600, facecolor='white')

# 绘制小提琴图
sns.violinplot(x='Group', y='Bias', data=data, bw_adjust=.5, cut=0, linewidth=1, palette="tab10",
inner_kws=dict(box_width=8, whis_width=1, color="0.3"))

# 使用盒子图来增强四分位距条
# sns.boxplot(x='Group', y='Bias', data=data, whis=np.inf, linewidth=2.5, palette="Set3")

# 设置 y 轴范围
ax.set(ylim=(-120, 30))

ax.set_yticks(np.arange(-120, 30, 10))  # 从-150到30，每10一横
ax.set_yticklabels([f'{tick:.1f}' for tick in np.arange(-120, 30, 10)])  # 设置y轴刻度标签
ax.set_title(f"bias comparison {case_type}", fontsize=12)
# 去除边框
# sns.despine(left=True, bottom=True)
plt.savefig(f"violin_bias_{case_type}.png", bbox_inches='tight', pad_inches=0.1)
# _meiyu
# _typhoon
# 显示图像
plt.show()
#%%
################################################################
# //
# //                       _oo0oo_
# //                      o8888888o
# //                      88" . "88
# //                      (| -_- |)
# //                      0\  =  /0
# //                    ___/`---'\___
# //                  .' \\|     |// '.
# //                 / \\|||  :  |||// \
# //                / _||||| -:- |||||- \
# //               |   | \\\  -  /// |   |
# //               | \_|  ''\---/''  |_/ |
# //               \  .-\__  '-'  ___/-. /
# //             ___'. .'  /--.--\  `. .'___
# //          ."" '<  `.___\_<|>_/___.' >' "".
# //         | | :  `- \`.;`\ _ /`;.`/ - ` : | |
# //         \  \ `_.   \_ __\ /__ _/   .-` /  /
# //     =====`-.____`.___ \_____/___.-`___.-'=====
# //                       `=---='
# //
# //
# //     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# //
# //               佛祖保佑         永無BUG
# //
#########################################################################
# // code出Bug了？
# // 　　　∩∩
# // 　　（´･ω･）
# // 　 ＿|　⊃／(＿＿
# // 　／ └-(＿＿＿／
# // 　￣￣￣￣￣￣￣
# // 算了反正不是我寫的
# // 　　 ⊂⌒／ヽ-、＿
# // 　／⊂/＿＿＿＿ ／
# // 　￣￣￣￣￣￣￣
# // 萬一是我寫的呢？
# // 　　　∩∩
# // 　　（´･ω･）
# // 　 ＿|　⊃／(＿＿
# // 　／ └-(＿＿＿／
# // 　￣￣￣￣￣￣￣
# // 算了反正改了一個又出三個
# // 　　 ⊂⌒／ヽ-、＿
# // 　／⊂/＿＿＿＿ ／
# // 　
########################################################################