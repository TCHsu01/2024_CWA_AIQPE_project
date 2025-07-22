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
from datetime import datetime, timedelta

from scipy import interpolate
import h5py
import xarray as xr
from scipy.interpolate import interp2d
import matplotlib.gridspec as gridspec

# set path
direct = 'D:/2024cwa/comparison/'
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

#########################################################################################
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
#########################################################################################
# # # # # All # # # # # #
# QPESUMS = QPESUMS_cb
# AI313 = AI313_cb
# AI322 = AI322_cb
# date = date_cb
########################################################################
# # # # # for general continuity case
start_dates = ['2023-05-01']
end_dates = [ '2024-07-28']
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
# case_type = 'typhoon'
# indices = get_indices(date_cb, case_type, start_dates, end_dates)

########################################################################
QPESUMS = QPESUMS_cb[indices]
AI313 = AI313_cb[indices]
AI322 = AI322_cb[indices]
date = date_cb[indices]


QPESUMS = QPESUMS.transpose((0, 2, 1))
# QPESUMS = QPESUMS[172:172+119]

AI322 = AI322.transpose((0, 2, 1))
# AI322 = AI322[172:172+119]

AI313 = AI313.transpose((0, 2, 1))
# AI313 = AI313[172:172+119]




mean_AI313  = np.nanmean(AI313 , axis=(1, 2))
mean_AI322 = np.nanmean(AI322, axis=(1, 2))
mean_QPESUMS = np.nanmean(QPESUMS, axis=(1, 2))



# Create the scatter plot
plt.figure(dpi=600,figsize=(8, 6),facecolor = 'white')
plt.scatter(mean_QPESUMS, mean_AI322, c='b', label='AI322')
plt.scatter(mean_QPESUMS, mean_AI313, c='#EAC100', label='AI313',alpha=0.5)

# Add y=x line
x = np.arange(-100,100,1)
y = x
plt.plot(x,y,color='k')

# Customize the plot
plt.xlabel('mean_QPESUMS')
plt.ylabel('mean_SPP')
plt.xlim(0,50)
plt.ylim(0,50)
plt.title(f'Scatter Plot for different SPP {case_type}',fontsize=18)
plt.grid()
plt.legend()

mean_bias_AI313 = np.nanmean(mean_AI313 - mean_QPESUMS)
mask = ~np.isnan(mean_QPESUMS) & ~np.isnan(mean_AI313)
correlation_AI313 = np.corrcoef(mean_QPESUMS[mask], mean_AI313[mask])[0, 1]
# 在右上方添加文本注释
plt.text(0.3, 0.85, f'Mean Bias_AI313: {mean_bias_AI313:.2f}', transform=plt.gca().transAxes, color='black')
plt.text(0.3, 0.80, f'Correlation_AI313: {correlation_AI313:.2f}', transform=plt.gca().transAxes, color='black')

mean_bias_AI322= np.nanmean(mean_AI322 - mean_QPESUMS)
mask = ~np.isnan(mean_QPESUMS) & ~np.isnan(mean_AI322)
correlation_AI322_STO = np.corrcoef(mean_QPESUMS[mask], mean_AI322[mask])[0, 1]


# 在右上方添加文本注释
plt.text(0.3, 0.95, f'Mean Bias_AI322: {mean_bias_AI322:.2f}', transform=plt.gca().transAxes, color='black')
plt.text(0.3, 0.9, f'Correlation_AI322: {correlation_AI322_STO:.2f}', transform=plt.gca().transAxes, color='black')

valid_filename = re.sub(r'[^\w\-_\. ]', '_', f"scatter_plot_for_different_SPP_{case_type}.png")
plt.savefig(valid_filename)  # 使用绝对路径

POD_thr_list = [0.5,2, 5, 10,15, 20,30, 50,80,120]

# MWCOMB = MW_PR
data = AI313

# 初始化空的结果数组
MW_PR_POD_list = []
MW_PR_FAR_list = []
MW_PR_CSI_list = []
MW_PR_BS_list = []
VHI_list = []
VFAR_list = []
VCSI_list = []
HSS_list = []
# 循环遍历不同的 POD_thr 值
for POD_thr in POD_thr_list:
    MW_PR_POD_H = np.zeros((len(data), 30, 40))
    MW_PR_POD_M = np.zeros((len(data), 30, 40))
    MW_PR_POD_F = np.zeros((len(data), 30, 40))
    MW_PR_POD_CN = np.zeros((len(data), 30, 40))        
    
    VHI_H = np.full((len(data), 30, 40), np.nan)
    VHI_M = np.full((len(data), 30, 40), np.nan)
    VHI_F = np.full((len(data), 30, 40), np.nan)
    
    MW_PR_POD_H[(QPESUMS > POD_thr) & (data > POD_thr)] = 1
    MW_PR_POD_M[(QPESUMS > POD_thr) & (data <= POD_thr)] = 1
    MW_PR_POD_F[(QPESUMS <= POD_thr) & (data > POD_thr)] = 1
    
    MW_PR_POD_H = np.nansum(MW_PR_POD_H, axis=0)
    MW_PR_POD_M = np.nansum(MW_PR_POD_M, axis=0)
    MW_PR_POD_F = np.nansum(MW_PR_POD_F, axis=0)
    MW_PR_POD_CN = np.nansum(MW_PR_POD_CN, axis=0)
    print(MW_PR_POD_H)
    MW_PR_POD = MW_PR_POD_H / (MW_PR_POD_H + MW_PR_POD_M)
    MW_PR_FAR = MW_PR_POD_F / (MW_PR_POD_H + MW_PR_POD_F)
    MW_PR_CSI = MW_PR_POD_H / (MW_PR_POD_H + MW_PR_POD_F + MW_PR_POD_M)
    MW_PR_BS = (MW_PR_POD_H + MW_PR_POD_F) / (MW_PR_POD_H + MW_PR_POD_M)    
    MW_PR_POD_N = MW_PR_POD_H + MW_PR_POD_F + MW_PR_POD_M + MW_PR_POD_CN
    MW_PR_HSS = 2*(MW_PR_POD_H*MW_PR_POD_N-MW_PR_POD_F*MW_PR_POD_M) / ((MW_PR_POD_H+MW_PR_POD_M)*(MW_PR_POD_M+MW_PR_POD_N)+(MW_PR_POD_H+MW_PR_POD_F)*(MW_PR_POD_F+MW_PR_POD_N))

    index = (QPESUMS > POD_thr) & (data > POD_thr)
    VHI_H[index] = data[index]
    VHI_H = np.nansum(VHI_H, axis=0)
    
    index2 = (QPESUMS > POD_thr) & (data <= POD_thr)
    VHI_M[index2] = QPESUMS[index2]
    VHI_M = np.nansum(VHI_M, axis=0)
    
    index3 = (QPESUMS <= POD_thr) & (data > POD_thr)
    VHI_F[index3] = data[index3]
    VHI_F = np.nansum(VHI_F, axis=0)
    
    VHI = VHI_H / (VHI_H + VHI_M)
    VFAR = VHI_F / (VHI_H + VHI_F)
    VCSI = VHI_H / (VHI_H + VHI_F + VHI_M)
    
    # 将每个 POD_thr 值下的结果添加到列表中
    MW_PR_POD_list.append(MW_PR_POD)
    MW_PR_FAR_list.append(MW_PR_FAR)
    MW_PR_CSI_list.append(MW_PR_CSI)
    MW_PR_BS_list.append(MW_PR_BS)
    HSS_list.append(MW_PR_HSS)
    VHI_list.append(VHI)
    VFAR_list.append(VFAR)
    VCSI_list.append(VCSI)

MW_PR_POD_list = np.asarray(MW_PR_POD_list)
MW_PR_FAR_list = np.asarray(MW_PR_FAR_list)
MW_PR_CSI_list = np.asarray(MW_PR_CSI_list)
MW_PR_BS_list = np.asarray(MW_PR_BS_list)
HSS_list = np.asarray(HSS_list)
VHI_list = np.asarray(VHI_list)
VFAR_list = np.asarray(VFAR_list)
VCSI_list = np.asarray(VCSI_list)

MW_PR_BS_list[np.isinf(MW_PR_BS_list)] = np.nan




MW_PR_POD_list1 = np.nanmean(MW_PR_POD_list,axis=(1,2))
MW_PR_FAR_list1 = np.nanmean(MW_PR_FAR_list,axis=(1,2))
MW_PR_CSI_list1 = np.nanmean(MW_PR_CSI_list,axis=(1,2))
MW_PR_BS_list1 = np.nanmean(MW_PR_BS_list,axis=(1,2))
HSS_list1 = np.nanmean(HSS_list,axis=(1,2))
VHI_list1 = np.nanmean(VHI_list,axis=(1,2))
VFAR_list1 = np.nanmean(VFAR_list,axis=(1,2))
VCSI_list1 = np.nanmean(VCSI_list,axis=(1,2))


POD_thr_list = [0.5,2, 5, 10,15, 20,30, 50,80,120]

# MWCOMB = MW_PR
data = AI322

# 初始化空的结果数组
MW_PR_POD_list = []
MW_PR_FAR_list = []
MW_PR_CSI_list = []
MW_PR_BS_list = []
VHI_list = []
VFAR_list = []
VCSI_list = []
HSS_list = []
# 循环遍历不同的 POD_thr 值
for POD_thr in POD_thr_list:
    MW_PR_POD_H = np.zeros((len(data),30, 40))
    MW_PR_POD_M = np.zeros((len(data),30, 40))
    MW_PR_POD_F = np.zeros((len(data),30, 40))
    MW_PR_POD_CN = np.zeros((len(data),30, 40))        
    
    VHI_H = np.full((len(data),30, 40), np.nan)
    VHI_M = np.full((len(data),30, 40), np.nan)
    VHI_F = np.full((len(data),30, 40), np.nan)
    
    MW_PR_POD_H[(QPESUMS > POD_thr) & (data > POD_thr)] = 1
    print(MW_PR_POD_H)
    MW_PR_POD_M[(QPESUMS > POD_thr) & (data <= POD_thr)] = 1
    MW_PR_POD_F[(QPESUMS <= POD_thr) & (data > POD_thr)] = 1
    
    MW_PR_POD_H = np.nansum(MW_PR_POD_H, axis=0)
    MW_PR_POD_M = np.nansum(MW_PR_POD_M, axis=0)
    MW_PR_POD_F = np.nansum(MW_PR_POD_F, axis=0)
    MW_PR_POD_CN = np.nansum(MW_PR_POD_CN, axis=0)
    # print(MW_PR_POD_H)
    MW_PR_POD = MW_PR_POD_H / (MW_PR_POD_H + MW_PR_POD_M)
    MW_PR_FAR = MW_PR_POD_F / (MW_PR_POD_H + MW_PR_POD_F)
    MW_PR_CSI = MW_PR_POD_H / (MW_PR_POD_H + MW_PR_POD_F + MW_PR_POD_M)
    MW_PR_BS = (MW_PR_POD_H + MW_PR_POD_F) / (MW_PR_POD_H + MW_PR_POD_M)    
    MW_PR_POD_N = MW_PR_POD_H + MW_PR_POD_F + MW_PR_POD_M + MW_PR_POD_CN
    MW_PR_HSS = 2*(MW_PR_POD_H*MW_PR_POD_N-MW_PR_POD_F*MW_PR_POD_M) / ((MW_PR_POD_H+MW_PR_POD_M)*(MW_PR_POD_M+MW_PR_POD_N)+(MW_PR_POD_H+MW_PR_POD_F)*(MW_PR_POD_F+MW_PR_POD_N))

    index = (QPESUMS > POD_thr) & (data > POD_thr)
    VHI_H[index] = data[index]
    VHI_H = np.nansum(VHI_H, axis=0)
    
    index2 = (QPESUMS > POD_thr) & (data <= POD_thr)
    VHI_M[index2] = QPESUMS[index2]
    VHI_M = np.nansum(VHI_M, axis=0)
    
    index3 = (QPESUMS <= POD_thr) & (data > POD_thr)
    VHI_F[index3] = data[index3]
    VHI_F = np.nansum(VHI_F, axis=0)
    
    VHI = VHI_H / (VHI_H + VHI_M)
    VFAR = VHI_F / (VHI_H + VHI_F)
    VCSI = VHI_H / (VHI_H + VHI_F + VHI_M)
    
    # 将每个 POD_thr 值下的结果添加到列表中
    MW_PR_POD_list.append(MW_PR_POD)
    MW_PR_FAR_list.append(MW_PR_FAR)
    MW_PR_CSI_list.append(MW_PR_CSI)
    MW_PR_BS_list.append(MW_PR_BS)
    HSS_list.append(MW_PR_HSS)
    VHI_list.append(VHI)
    VFAR_list.append(VFAR)
    VCSI_list.append(VCSI)

MW_PR_POD_list2 = np.asarray(MW_PR_POD_list)
MW_PR_FAR_list2 = np.asarray(MW_PR_FAR_list)
MW_PR_CSI_list2 = np.asarray(MW_PR_CSI_list)
MW_PR_BS_list2 = np.asarray(MW_PR_BS_list)
HSS_list2 = np.asarray(HSS_list)
VHI_list2 = np.asarray(VHI_list)
VFAR_list2 = np.asarray(VFAR_list)
VCSI_list2 = np.asarray(VCSI_list)

MW_PR_BS_list2[np.isinf(MW_PR_BS_list2)] = np.nan


MW_PR_POD_list2 = np.nanmean(MW_PR_POD_list2,axis=(1,2))
MW_PR_FAR_list2 = np.nanmean(MW_PR_FAR_list2,axis=(1,2))
MW_PR_CSI_list2 = np.nanmean(MW_PR_CSI_list2,axis=(1,2))
MW_PR_BS_list2 = np.nanmean(MW_PR_BS_list2,axis=(1,2))
HSS_list2 = np.nanmean(HSS_list2,axis=(1,2))
VHI_list2 = np.nanmean(VHI_list2,axis=(1,2))
VFAR_list2 = np.nanmean(VFAR_list2,axis=(1,2))
VCSI_list2 = np.nanmean(VCSI_list2,axis=(1,2))


fig, ax1 = plt.subplots(dpi=600, figsize=(10, 6),facecolor = 'white')

# 绘制 SPP skill score 数据（POD, FAR, CSI）
ax1.plot(POD_thr_list, MW_PR_POD_list1, label='POD_AI313', marker='o', color='r', linestyle='--')
ax1.plot(POD_thr_list, MW_PR_FAR_list1, label='FAR_AI313', marker='o', color='b', linestyle='--')
ax1.plot(POD_thr_list, MW_PR_CSI_list1, label='CSI_AI313', marker='o', color='k', linestyle='--',linewidth = 2)
ax1.plot(POD_thr_list, MW_PR_BS_list1, label='BS_AI313', marker='o', color='y', linestyle='--')
ax1.plot(POD_thr_list, MW_PR_POD_list2, label='POD_AI322', marker='o', color='r')
ax1.plot(POD_thr_list, MW_PR_FAR_list2, label='FAR_AI322', marker='o', color='b')
ax1.plot(POD_thr_list, MW_PR_CSI_list2, label='CSI_AI322', marker='o', color='k')
ax1.plot(POD_thr_list, MW_PR_BS_list2, label='BS_AI322', marker='o', color='y')
plt.ylim(0,1.4)
ax1.set_xlabel('Threshold (mm/day)')
ax1.set_ylabel('Value')
ax1.set_title(f'SPP skill score w.r.t. different threshold {case_type}', fontsize=14)
ax1.legend(loc='upper right',ncol = 2)
ax1.grid()
valid_filename = re.sub(r'[^\w\-_\. ]', '_', f"SPP_skill_score_{case_type}.png")
plt.savefig(valid_filename)  # 使用绝对路径


