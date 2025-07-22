###### 說明區 ######
# 東西有點多有點亂，要耐心處理
# 把做出來的npy檔案讀進來之後可以畫以下的圖
# 1. 每日累積降雨圖的三種產品比對圖
# 2. 計算降雨強度的空間分布（藍色的馬賽克圖）
# 3. 沒那麼好看的density scatter plot（好看的在ver2裡面）
# 4. 空間上的降雨相關係數圖
# 5. 小提琴圖
###################
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
import sys


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
# direct = 'H:/2024cwa/comparison/'
# file_list = os.listdir(direct)

# QPESUMS_23 = np.load(os.path.join(direct, 'new_QPESUMS_daily_20230501_20230831.npy'))
# QPESUMS_24 = np.load(os.path.join(direct, 'new_QPESUMS_daily_20240501_20240630.npy'))
# QPESUMS_GM = np.load(os.path.join(direct, 'new_QPESUMS_daily_20240721_20240728.npy'))
# QPESUMS_cb = np.concatenate((QPESUMS_23, QPESUMS_24, QPESUMS_GM), axis=0)

# AI313_23 = np.load(os.path.join(direct, 'island_313_degrade_aiqpe_20230501_20230831.npy'))
# AI313_24 = np.load(os.path.join(direct, 'island_313_degrade_aiqpe_20240501_20240630.npy'))
# AI313_GM = np.load(os.path.join(direct, 'island_313_degrade_aiqpe_20240721_20240728.npy'))
# AI313_cb = np.concatenate((AI313_23, AI313_24, AI313_GM), axis=0)

# AI322_23 = np.load(os.path.join(direct, 'island_322_degrade_aiqpe_20230501_20230831.npy'))
# AI322_24 = np.load(os.path.join(direct, 'island_322_degrade_aiqpe_20240501_20240630.npy'))
# AI322_GM = np.load(os.path.join(direct, 'island_322_degrade_aiqpe_20240721_20240728.npy'))
# AI322_cb = np.concatenate((AI322_23, AI322_24, AI322_GM), axis=0)

# date_23 = np.load(os.path.join(direct, 'date_list_20230501_20230831.npy'))
# date_24 = np.load(os.path.join(direct, 'date_list_20240501_20240630.npy'))
# date_GM = np.load(os.path.join(direct, 'date_list_20240721_20240728.npy'))
# date_cb = np.concatenate((date_23, date_24, date_GM), axis=0)


# QPESUMS = QPESUMS_cb
# AI313 = AI313_cb
# AI322 = AI322_cb
# date = date_cb
######################################################################################
# # # # # # # for mayu case
# start_date = datetime.strptime('2023-05-01', '%Y-%m-%d')
# end_date = datetime.strptime('2023-06-30', '%Y-%m-%d')
# x = np.array([datetime.strptime(date_str, '%Y-%m-%d') for date_str in date_cb])
# indices23 = [i for i, t in enumerate(x) if start_date <= t <= end_date]
# start_date = datetime.strptime('2024-05-01', '%Y-%m-%d')
# end_date = datetime.strptime('2024-06-30', '%Y-%m-%d')
# x = np.array([datetime.strptime(date_str, '%Y-%m-%d') for date_str in date_cb])
# indices24 = [i for i, t in enumerate(x) if start_date <= t <= end_date]
# indices = np.concatenate([indices23,indices24])

# start_date = datetime.strptime('2023-05-01', '%Y-%m-%d')
# end_date = datetime.strptime('2024-07-28', '%Y-%m-%d')
# x = np.array([datetime.strptime(date_str, '%Y-%m-%d') for date_str in date_cb])
# indices = [i for i, t in enumerate(x) if start_date <= t <= end_date]
# ####################################################################################
#########################################################################################
# # # # # TC cases # # # # # #
# set path
direct = 'D:/2024cwa/AIQPE_code_2024'
file_list = os.listdir(direct)
def link(file1,file2,file3, file4, file5):
    return np.concatenate((file1, file2, file3, file4, file5))

QPESUMS = np.load(os.path.join(direct, 'new_QPESUMS_daily_20230529_20230531.npy'))
QPESUMS_2 = np.load(os.path.join(direct, 'new_QPESUMS_daily_20230724_20230728.npy'))
QPESUMS_3 = np.load(os.path.join(direct, 'new_QPESUMS_daily_20230801_20230804.npy'))
QPESUMS_4 = np.load(os.path.join(direct, 'new_QPESUMS_daily_20230828_20230831.npy'))
QPESUMS_gaemi = np.load(os.path.join(direct, 'new_QPESUMS_daily_20240721_20240728.npy'))
QPESUMS_gaemi =  QPESUMS_gaemi[1:6]
print(np.shape(QPESUMS_gaemi))

AI313 = np.load(os.path.join(direct, 'island_313_degrade_aiqpe_20230529_20230531.npy'))
AI313_2 = np.load(os.path.join(direct, 'island_313_degrade_aiqpe_20230724_20230728.npy'))
AI313_3 = np.load(os.path.join(direct, 'island_313_degrade_aiqpe_20230801_20230804.npy'))
AI313_4 = np.load(os.path.join(direct, 'island_313_degrade_aiqpe_20230828_20230831.npy'))
AI313_gaemi = np.load(os.path.join(direct, 'island_313_degrade_aiqpe_20240722_20240728.npy'))
AI313_gaemi = AI313_gaemi[0:5]

AI322 = np.load(os.path.join(direct, 'island_322_degrade_aiqpe_20230529_20230531.npy'))
AI322_2 = np.load(os.path.join(direct, 'island_322_degrade_aiqpe_20230724_20230728.npy'))
AI322_3 = np.load(os.path.join(direct, 'island_322_degrade_aiqpe_20230801_20230804.npy'))
AI322_4 = np.load(os.path.join(direct, 'island_322_degrade_aiqpe_20230828_20230831.npy'))
AI322_gaemi = np.load(os.path.join(direct, 'island_322_degrade_aiqpe_20240722_20240728.npy'))
AI322_gaemi = AI322_gaemi[0:5]

date = np.load(os.path.join(direct, 'date_list_20230529_20230531.npy'))
date_2 = np.load(os.path.join(direct, 'date_list_20230724_20230728.npy'))
date_3 = np.load(os.path.join(direct, 'date_list_20230801_20230804.npy'))
date_4 = np.load(os.path.join(direct, 'date_list_20230828_20230831.npy'))
date_gaemi = np.load(os.path.join(direct, 'date_list_20240722_20240728.npy'))
date_gaemi = date_gaemi[0:5]

QPESUMS_all = link(QPESUMS, QPESUMS_2, QPESUMS_3, QPESUMS_4, QPESUMS_gaemi)
AI313_all = link(AI313, AI313_2, AI313_3, AI313_4, AI313_gaemi)
AI322_all = link(AI322, AI322_2, AI322_3, AI322_4, AI322_gaemi)
date_all = link(date, date_2, date_3, date_4, date_gaemi)

QPESUMS = QPESUMS_all
AI313 = AI313_all
AI322 = AI322_all
date = date_all
indices = np.arange(0,len(QPESUMS),1)
########################################################################################
data = pd.DataFrame({'AI313': AI313[indices].flatten(),
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

theLons = np.arange(120, 122.99, 0.1)##
theLats = np.arange(22, 25.99, 0.1)##
#%%
clevs, cmaplist = setcolorQPF(cnum=2)
#cmap = (mpl.colors.ListedColormap(cmaplist/255.).with_extremes(
#        over=(253/255., 201/255., 255/255.), under='white'))
# 修改的地方開始
cmap = mpl.colors.ListedColormap(cmaplist / 255.)
cmap.set_over((253 / 255., 201 / 255., 255 / 255.))
cmap.set_under('white')
# 修改的地方結束
norm = mpl.colors.BoundaryNorm(clevs, cmap.N)
clevs_labels = [str(v) if isinstance(v, float) else str(int(v)) for v in clevs]
# part 1. along time
along_time_coeff_AI313 = [];along_time_bias_AI313 = [];mean_aipqe_AI313 = []
along_time_coeff_AI322 = [];along_time_bias_AI322 = [];mean_aipqe_AI322 = []
mean_qpesums = []

for i in range(len(indices)):
    formatted_date = date[indices[i]]                   # 日期
    #######
    data = pd.DataFrame({'AI313 ': AI313[indices[i]].flatten(),'QPESUMS': QPESUMS[indices[i]].flatten()})
    coef = data.corr();coef = np.asarray(coef)
    time_daily_corr1 = coef[0, 1]
    time_bias1 = AI313[i] - QPESUMS[i];time_bias1 = np.nanmean(time_bias1)
    sumqpe1 = np.nanmean(AI313[indices[i]])
    along_time_coeff_AI313.append(time_daily_corr1)
    along_time_bias_AI313.append(time_bias1)
    mean_aipqe_AI313.append(sumqpe1)
    #######
    data = pd.DataFrame({'AI322 ': AI322[indices[i]].flatten(),'QPESUMS': QPESUMS[indices[i]].flatten()})
    coef = data.corr();coef = np.asarray(coef)
    time_daily_corr2 = coef[0, 1]
    time_bias2 = AI322[i] - QPESUMS[i];time_bias2 = np.nanmean(time_bias2)
    sumqpe2 = np.nanmean(AI322[indices[i]])
    along_time_coeff_AI322.append(time_daily_corr2)
    along_time_bias_AI322.append(time_bias2)
    mean_aipqe_AI322.append(sumqpe2)
    
    sumqpesum = np.nanmean(QPESUMS[indices[i]])
    mean_qpesums.append(sumqpesum)
    ###########################################################################################
    # 出圖的部分
    ###########################################################################################
    
    fig, axs = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True, facecolor='white')
    m = Basemap(projection='cyl', resolution='i', fix_aspect=True,
        llcrnrlon=120, llcrnrlat=22, urcrnrlon=122.5, urcrnrlat=25.5, lat_ts=22)
    # m = Basemap(projection='cyl', resolution='i', fix_aspect=True,
    #             llcrnrlon=118, llcrnrlat=19.99, urcrnrlon=123.5, urcrnrlat=26.99, lat_ts=20)

    # 第一張：QPESUM
    ax = axs[0]
    cx, cy = np.meshgrid(theLons, theLats)
    c = m.contourf(cx, cy, QPESUMS[indices[i]], clevs, cmap=cmap, norm=norm, ax=ax)
    m.drawparallels(np.arange(22, 25.5, 2), labels=[1, 0, 0, 0], color='#787878', fontsize=12, ax=ax)
    m.drawmeridians(np.arange(120, 122.5, 2), labels=[0, 1, 0, 1], color='#787878', fontsize=12, ax=ax)
    m.drawcoastlines(color='#28FF28', ax=ax)
    
    ax.set_title(f"QPESUM for {date[indices[i]]}", fontsize=12)

    # 第二張：AI 313
    ax = axs[1]
    cx, cy = np.meshgrid(theLons, theLats)
    c = m.contourf(cx, cy, AI313[indices[i]], clevs, cmap=cmap, norm=norm, ax=ax)
    m.drawparallels(np.arange(22, 25.5, 2), labels=[1, 0, 0, 0], color='#787878', fontsize=12, ax=ax)
    m.drawmeridians(np.arange(120, 122.5, 2), labels=[0, 1, 0, 1], color='#787878', fontsize=12, ax=ax)
    m.drawcoastlines(color='#28FF28', ax=ax)
    axs[1].annotate(f'CC: {time_daily_corr1:.2f}\nBias: {time_bias1:.2f}', xy=(0.97, 0.97), 
                xycoords='axes fraction', horizontalalignment='right', verticalalignment='top',
                bbox=dict(facecolor='white', alpha=0.8))
    ax.set_title(f"AI 313 for {date[indices[i]]}", fontsize=12)

    # 第三張：AI 322
    ax = axs[2]
    c = m.contourf(cx, cy, AI322[indices[i]], clevs, cmap=cmap, norm=norm, ax=ax)
    m.drawparallels(np.arange(22, 25.5, 2), labels=[1, 0, 0, 0], color='#787878', fontsize=12, ax=ax)
    m.drawmeridians(np.arange(120, 122.5, 2), labels=[0, 1, 0, 1], color='#787878', fontsize=12, ax=ax)
    m.drawcoastlines(color='#28FF28', ax=ax)
    axs[2].annotate(f'CC: {time_daily_corr2:.2f}\nBias: {time_bias2:.2f}', xy=(0.97, 0.97), 
                xycoords='axes fraction', horizontalalignment='right', verticalalignment='top',
                bbox=dict(facecolor='white', alpha=0.8))
    ax.set_title(f"AI 322 for {date[indices[i]]}", fontsize=12)
    # 共用 colorbar
    fig.colorbar(c, ax=axs, orientation='vertical', fraction=.02, pad=0.04).set_label('Rainfall rate(mm/hr)')
    
    valid_filename = re.sub(r'[^\w\-_\. ]', '_', f"island_comparison_{formatted_date}.png")
    plt.savefig(os.path.join(direct, valid_filename))  # 使用绝对路径
    # plt.savefig(valid_filename)
    plt.show()

#%%
# midium_coeff_313 = np.nanmedian(along_time_coeff_AI313, axis=None, out=None, overwrite_input=False, keepdims=False)
# midium_coeff_322 = np.nanmedian(along_time_coeff_AI322, axis=None, out=None, overwrite_input=False, keepdims=False)
# axx = np.arange(1,193)
# plt.scatter(mean_qpesums,mean_aipqe_AI313,alpha=0.5,label='AIQPE313')
# plt.scatter(mean_qpesums,mean_aipqe_AI322,alpha=0.5,label='AIQPE322')
# plt.xlabel('QPESUMS')
# plt.ylabel('AI IRQPE')
# plt.legend()
# x = np.arange(0,1000,1)
# plt.plot(x,x)
# plt.xlim(0,100);plt.ylim(0,100)

# ######
# flat_QPESUMS_cb = QPESUMS_cb.flatten() 
# flat_AI313_cb = AI313_cb.flatten()
# flat_AI322_cb = AI322_cb.flatten()
# data_flat = pd.DataFrame({'AI 322':flat_AI322_cb,
#                          'QPESUMS': flat_QPESUMS_cb,
#                          'AI 313':flat_AI313_cb})
# data_flat_clean = data_flat.replace([np.inf, -np.inf], np.nan).dropna()
# plt.figure(dpi=600, facecolor='white')
# plt.scatter(flat_QPESUMS_cb,flat_AI313_cb,alpha=0.5,label='AIQPE313',s=1)
# plt.plot(x,x);plt.xlim(0,800);plt.ylim(0,800)
# plt.xlabel('QPESUMS');plt.ylabel('AI IRQPE 313')
# plt.show()

# plt.figure(dpi=600, facecolor='white')

# # sns.kdeplot(x=data_flat['QPESUMS'], y=data_flat['AI 322'],fill = "true",
# #             cmap = 'Blues',cbar = "true")
# plt.scatter(flat_QPESUMS_cb,flat_AI322_cb,alpha=0.5,label='AIQPE322',s=1)
# # plt.plot(x,x);plt.xlim(0,800);plt.ylim(0,800)
# # # plt.xlabel('QPESUMS');plt.ylabel('AI IRQPE 322')
# # plt.show()
# sns.kdeplot(x=data_flat['QPESUMS'], y=data_flat['AI 322'], fill=True,
#             cmap='Blues', cbar=True)
# plt.plot(x, x, color='red')
# plt.xlim(0, 800)
# plt.ylim(0, 800)
# plt.xlabel('QPESUMS')
# plt.ylabel('AI IRQPE 322')
# plt.show()


# Density scatter plot 313:沒分範圍
# fig, ax = plt.subplots(dpi=600, facecolor='white')
# xy = np.vstack([data_flat_clean['QPESUMS'], data_flat_clean['AI 313']]);z = gaussian_kde(xy)(xy)
# sc = ax.scatter(data_flat_clean['QPESUMS'], data_flat_clean['AI 313'], c=z, s=1, cmap='viridis')
# diff = np.subtract(data_flat_clean['QPESUMS'],  data_flat_clean['AI 313'])
# square = np.square(diff);MSE = square.mean();RMSE = np.sqrt(MSE)
# plt.plot(x, x, color='red')
# ax.set_xscale('log')
# ax.set_yscale('log')
# plt.xlim(0.5, 800);plt.ylim(0.5, 800)
# plt.xlabel('QPESUMS');plt.ylabel('AI IRQPE 313')
# plt.title('Density scatter plot of QPESUMS and AI_IRQPE_313')
# #====================================================================================================================================
# # 计算相关系数
# correlation_coefficient = np.corrcoef(data_flat_clean['QPESUMS'], data_flat_clean['AI 313'])[0, 1]
# # 添加颜色条
# cbar = plt.colorbar(sc)
# cbar.set_label('Density')
#
# # 在图上添加 RMSE 和相关系数
# ax.annotate(f'RMSE: {RMSE:.2f}\ncorr: {correlation_coefficient:.2f}',
#             xy=(0.05, 0.95),
#             xycoords='axes fraction',
#             fontsize=12,
#             color='black',
#             horizontalalignment='left',
#             verticalalignment='top',
#             bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
# plt.savefig(os.path.join('C:/Users/haapy meow/Desktop/Topics/CWA/專題/AIQPE_code_2024/CC_pic/', "AI_IRQPE_313_QPESUM_all_flat_comparison.png"))  # 使用绝对路径
# plt.show()
# ######
# fig, ax = plt.subplots(dpi=600, facecolor='white')
# # Dessity scatter plot 322:沒分範圍
# xy = np.vstack([data_flat_clean['QPESUMS'], data_flat_clean['AI 322']]);z = gaussian_kde(xy)(xy)
# sc = ax.scatter(data_flat_clean['QPESUMS'], data_flat_clean['AI 322'], c=z, s=1, cmap='viridis')
# diff = np.subtract(data_flat_clean['QPESUMS'],  data_flat_clean['AI 322'])
# square = np.square(diff);MSE = square.mean();RMSE = np.sqrt(MSE)
# plt.plot(x, x, color='red')
# ax.set_xscale('log')
# ax.set_yscale('log')
# # plt.xlim(min_val, 800);plt.ylim(min_val, 800)
# plt.xlabel('QPESUMS');plt.ylabel('AI IRQPE 322')
# plt.title('Density scatter plot of QPESUMS and AI_IRQPE_322')
#
# # 计算相关系数
# correlation_coefficient = np.corrcoef(data_flat_clean['QPESUMS'], data_flat_clean['AI 322'])[0, 1]
# # 添加颜色条
# cbar = plt.colorbar(sc)
# cbar.set_label('Density')
#
# # 在图上添加 RMSE 和相关系数
# ax.annotate(f'RMSE: {RMSE:.2f}\ncorr: {correlation_coefficient:.2f}',
#             xy=(0.05, 0.95),
#             xycoords='axes fraction',
#             fontsize=12,
#             color='black',
#             horizontalalignment='left',
#             verticalalignment='top',
#             bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
# plt.savefig(os.path.join('C:/Users/haapy meow/Desktop/Topics/CWA/專題/AIQPE_code_2024/CC_pic/', "AI_IRQPE_322_QPESUM_all_flat_comparison.png"))  # 使用绝对路径
# plt.show()
# #====================================================================================================================================
#%%
#分大小範圍-降雨格點數大於每日的中位數為大範圍降雨
wide_qpe = []; wide_313 = []; wide_322 = []
local_qpe = []; local_313 = []; local_322 = []

AI313_flat_nonan_percent = []
AI322_flat_nonan_percent = []
QPESUMS_flat_nonan_percent = []

AI322_flat = AI322[indices].reshape(-1,1200)
QPESUMS_flat = QPESUMS[indices].reshape(-1,1200)
AI313_flat = AI313[indices].reshape(-1,1200) 

# #中位數
# median_AI322 = np.nanmedian(AI322_flat, axis=1)
# median_QPESUMS = np.nanmedian(QPESUMS_flat, axis=1)
# median_AI313 = np.nanmedian(AI313_flat, axis=1)

AI313_flat_nonan = np.nan_to_num(AI313_flat, nan=0.0)
AI322_flat_nonan = np.nan_to_num(AI322_flat, nan=0.0)
QPESUMS_flat_nonan = np.nan_to_num(QPESUMS_flat, nan=0.0)

AI313_flat_nonan[AI313_flat_nonan < 0.5] = 0
AI322_flat_nonan[AI322_flat_nonan < 0.5] = 0
QPESUMS_flat_nonan[QPESUMS_flat_nonan < 0.5] = 0


a = 0
# m = Basemap(projection='cyl', resolution='i', fix_aspect=True,
#             llcrnrlon=120, llcrnrlat=22.0, urcrnrlon=122.99, urcrnrlat=25.99, lat_ts=22.0)

# latitudes = np.arange(22.0, 25.99, 0.1)
# longitudes = np.arange(120.0, 122.99, 0.1)
# land_count = 0
# for lat in latitudes:
#     for lon in longitudes:
#         if m.is_land(lon, lat):
#             land_count += 1
# print("属于陆地的网格点数量:", land_count)
# print("ok")
print("属于陆地的网格点数量:", 316)
q = []; AI13 = []; AI22 = []

# h=np.where(QPESUMS_flat_nonan>0)
# a = len(h[1])
# print(a)
# sys.exit()
wide_day = 0
local_day = 0
for i in range(QPESUMS_flat.shape[0]):
    valid_count_QPESUMS = np.count_nonzero((QPESUMS_flat_nonan[i]))
    valid_count_AI313 = np.count_nonzero(AI313_flat_nonan[i])
    valid_count_AI322 = np.count_nonzero((AI322_flat_nonan[i]))
    if valid_count_QPESUMS < 158 :
        local_day += 1
        local_qpe.append(QPESUMS_flat_nonan[i])
        local_313.append(AI313_flat_nonan[i])
        local_322.append(AI322_flat_nonan[i])
    else:
        wide_day += 1
        wide_313.append(AI313_flat_nonan[i])
        wide_qpe.append(QPESUMS_flat_nonan[i])
        wide_322.append(AI322_flat_nonan[i])

    # q.append(str(i) + ' ' + str(valid_count_QPESUMS))
    # AI13.append(str(i) + ' ' + str(valid_count_AI313))
    # AI22.append(str(i) + ' ' + str(valid_count_AI322))
    a+=1
    
    # AI313_flat_nonan_percent[i] = np.round((valid_count_AI313 / 316)*100, 2)
    # AI322_flat_nonan_percent[i] =  np.round((valid_count_AI322 / 316)*100, 2)
    # QPESUMS_flat_nonan_percent[i] = np.round((valid_count_QPESUMS / 316)*100, 2)


    ai313 = np.round((valid_count_AI313 / 316)*100, 2)
    ai322 = np.round((valid_count_AI322 / 316)*100, 2)
    qpesums = np.round((valid_count_QPESUMS / 316)*100, 2)
    AI313_flat_nonan_percent.append(ai313)
    AI322_flat_nonan_percent.append(ai322)
    QPESUMS_flat_nonan_percent.append(qpesums)

    print(valid_count_QPESUMS,QPESUMS_flat_nonan_percent[i],"%")
    print(valid_count_AI313,AI313_flat_nonan_percent[i],"%")
    print(valid_count_AI322,AI322_flat_nonan_percent[i],"%")

    print("date:", a)
    print("=======================")
    
    # if np.sum(~np.isnan(AI322_flat[i])) >= 600:
    #     wide_322.append(AI322_flat[i])
    #     a+=1
    # else:
    #     local_322.append(AI322_flat[i])
    #     b+=1
    # if  np.sum(~np.isnan(AI313_flat[i])) >= 600:
    #     wide_313.append(AI313_flat[i])
    # else:
    #     local_313.append(AI313_flat[i])
    # if np.sum(~np.isnan(QPESUMS_flat[i])) >= 600:
    #     wide_qpe.append(QPESUMS_flat[i])
    # else:
    #     local_qpe.append(QPESUMS_flat[i])
    # for j in range(QPESUMS_flat.shape[1]):
    #     if AI313_flat_nonan[i][j] > 0:
    #         a+=1
    #     else:
    #         b+=1
    # print("============================")

# with open('output.txt', 'w') as f:
#     for item in q, AI13, AI22:
#         f.write("%s\n" % item)
#print("Data has been written to 'output.txt'")
    # print("============================")
print("wide:", len(wide_322), len(wide_313), len(wide_qpe))
print("local:", len(local_322), len(local_313), len(local_qpe))
print("wide day :", wide_day)
print("local day :", local_day)
print(np.shape(QPESUMS_flat_nonan_percent))
print(np.shape(AI313_flat_nonan_percent))
print(np.shape(AI322_flat_nonan_percent))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
# 创建自定义颜色映射
def manual_cmap():
    colors = [
        [220,220,220],  # 灰色
        [187,217,234],
        [155,192,215],
        [123,165,194],
        [92,139,174],
        [60,112,153],
        [47,98,141],
        [44,91,134],
        [41,83,128],
        [38,76,121],
        [36,69,114],
        [33,62,107],
        [30,54,101],
        [27,47,94],
        [24,40,87],
        [22,33,81],
    ]
    return mcolors.LinearSegmentedColormap.from_list("manual_cmap", np.array(colors) / 255)

# 设置颜色映射和归一化
cmap = manual_cmap()
norm = mcolors.BoundaryNorm(boundaries=np.linspace(0, 14, 15), ncolors=cmap.N, clip=True)

# 绘制散点图和热图，并将计数转换为百分比
def plot_heatmap(ax, x, y, xlabel, ylabel, title):
    hist, xedges, yedges = np.histogram2d(x, y, bins=(10, 10), range=[[0, 100], [0, 100]])
    hist_percent = (hist / hist.sum()) * 100
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    im = ax.imshow(hist_percent.T, extent=extent, origin='lower', cmap=cmap, norm=norm, aspect='auto')
    ax.scatter(x, y, color='red', marker='o', alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid()
    return im
# 创建图形和子图
plt.figure(dpi=600,figsize=(10, 15),facecolor='white')
fig, axs = plt.subplots(1, 2, figsize=(12, 5), dpi=600, facecolor='white')

# 绘制两个子图
im1 = plot_heatmap(axs[0], QPESUMS_flat_nonan_percent, AI313_flat_nonan_percent,
 'QPESUMS percentage (%)', 'AI313 percentage (%)', 'Rain area compare with 313 typhoon')
im2 = plot_heatmap(axs[1], QPESUMS_flat_nonan_percent, AI322_flat_nonan_percent,
 'QPESUMS percentage (%)', 'AI322 percentage (%)', 'Rain area compare with 322 typhoon')

# 添加一个共用的颜色条
cbar = fig.colorbar(im1, ax=axs, orientation='vertical', fraction=0.02, pad=0.04, ticks=np.linspace(0, 14, 15))
cbar.set_label('percentage (%)')
plt.savefig("mosaic_heatmap_comparison_typhoon.png")  # 使用绝对路径
# 显示图表
plt.show()
# # 绘制散点图
axs[0].scatter(QPESUMS_flat_nonan_percent, AI313_flat_nonan_percent, color='blue', marker='o')
# s设置标题和标签
axs[0].set_title('Rain area compare with 313')
axs[0].set_xlabel('QPESUMS percentage (%)');axs[0].set_ylabel('AI313 percentag (%)')
axs[0].grid()

axs[1].scatter(QPESUMS_flat_nonan_percent, AI322_flat_nonan_percent, color='blue', marker='o')
# 设置标题和标签
axs[1].set_title('Rain area compare with 322')
axs[1].set_xlabel('QPESUMS percent (%)');axs[1].set_ylabel('AI322 percent (%)')
axs[1].grid()

# 显示图表
plt.show()
# sys.exit()
#========================================================================
#%%
wide_qpe = np.array(wide_qpe); wide_313 = np.array(wide_313); wide_322 = np.array(wide_322)
local_qpe = np.array(local_qpe); local_313 = np.array(local_313); local_322 = np.array(local_322)
#print(np.shape(local_qpe))

# # Dessity scatter plot 313大範圍
fig, ax = plt.subplots(dpi=600, facecolor='white')
wide =  pd.DataFrame({'AI322': wide_322.flatten(),
                      'QPESUMS': wide_qpe.flatten(),
                      'AI313': wide_313.flatten()})
local = pd.DataFrame({'AI322': local_322.flatten(),
                      'QPESUMS': local_qpe.flatten(),
                      'AI313': local_313.flatten()})
wide_clean = wide.replace([np.inf, -np.inf], np.nan).dropna()
local_clean = local.replace([np.inf, -np.inf], np.nan).dropna()
xy = np.vstack([wide_clean['QPESUMS'], wide_clean['AI313']]);z = gaussian_kde(xy)(xy)
sc = ax.scatter(wide_clean['QPESUMS'], wide_clean['AI313'], c=z, s=1, cmap='viridis')
diff = np.subtract(wide_clean['QPESUMS'], wide_clean['AI313'])
square = np.square(diff);MSE = square.mean();RMSE = np.sqrt(MSE)
plt.plot(x, x, color='red')
ax.set_xscale('log')
ax.set_yscale('log')
# plt.xlim(min_val, 800);plt.ylim(, 800)
plt.xlabel('QPESUMS');plt.ylabel('AI IRQPE 313')
plt.title('Widespread precipitation for AI_IRQPE_313')

# 计算相关系数
correlation_coefficient = np.corrcoef(wide_clean['QPESUMS'], wide_clean['AI313'])[0, 1]
# 添加颜色条
cbar = plt.colorbar(sc)
cbar.set_label('Density')

# 在图上添加 RMSE 和相关系数
ax.annotate(f'RMSE: {RMSE:.2f}\ncorr: {correlation_coefficient:.2f}', 
            xy=(0.05, 0.95), 
            xycoords='axes fraction', 
            fontsize=12, 
            color='black',
            horizontalalignment='left', 
            verticalalignment='top',
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
# plt.savefig(os.path.join('C:/Users/haapy meow/Desktop/Topics/CWA/專題/AIQPE_code_2024/CC_pic/', "AI_IRQPE_313_QPESUM_all_flat_comparison.png"))  # 使用绝对路径
plt.show()

# Dessity scatter plot 322大範圍
fig, ax = plt.subplots(dpi=600, facecolor='white')
wide =  pd.DataFrame({'AI322': wide_322.flatten(),
                      'QPESUMS': wide_qpe.flatten(),
                      'AI313': wide_313.flatten()})
wide_clean = wide.replace([np.inf, -np.inf], np.nan).dropna()
xy = np.vstack([wide_clean['QPESUMS'], wide_clean['AI313']]);z = gaussian_kde(xy)(xy)
sc = ax.scatter(wide_clean['QPESUMS'], wide_clean['AI322'], c=z, s=1, cmap='viridis')
diff = np.subtract(wide_clean['QPESUMS'], wide_clean['AI322'])
square = np.square(diff);MSE = square.mean();RMSE = np.sqrt(MSE)
plt.plot(x, x, color='red')
ax.set_xscale('log')
ax.set_yscale('log')
# plt.xlim(min_val, 800);plt.ylim(, 800)
plt.xlabel('QPESUMS');plt.ylabel('AI IRQPE 322')
plt.title('Widespread precipitation for AI_IRQPE_322')

# 计算相关系数
correlation_coefficient = np.corrcoef(wide_clean['QPESUMS'], wide_clean['AI322'])[0, 1]
# 添加颜色条
cbar = plt.colorbar(sc)
cbar.set_label('Density')

# 在图上添加 RMSE 和相关系数
ax.annotate(f'RMSE: {RMSE:.2f}\ncorr: {correlation_coefficient:.2f}', 
            xy=(0.05, 0.95), 
            xycoords='axes fraction', 
            fontsize=12, 
            color='black',
            horizontalalignment='left', 
            verticalalignment='top',
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
# plt.savefig(os.path.join('C:/Users/haapy meow/Desktop/Topics/CWA/專題/AIQPE_code_2024/CC_pic/', "AI_IRQPE_313_QPESUM_all_flat_comparison.png"))  # 使用绝对路径
plt.show()
#################################################################################
# # Dessity scatter plot 313小範圍
#################################################################################
fig, ax = plt.subplots(dpi=600, facecolor='white')
local = pd.DataFrame({'AI322': local_322.flatten(),
                      'QPESUMS': local_qpe.flatten(),
                      'AI313': local_313.flatten()})
local_clean = local.replace([np.inf, -np.inf], np.nan).dropna()
xy = np.vstack([local_clean['QPESUMS'], local_clean['AI313']]);z = gaussian_kde(xy)(xy)
sc = ax.scatter(local_clean['QPESUMS'], local_clean['AI313'], c=z, s=1, cmap='viridis')
diff = np.subtract(local_clean['QPESUMS'], local_clean['AI313'])
square = np.square(diff);MSE = square.mean();RMSE = np.sqrt(MSE)
plt.plot(x, x, color='red')
ax.set_xscale('log')
ax.set_yscale('log')
# plt.xlim(min_val, 800);plt.ylim(, 800)
plt.xlabel('QPESUMS');plt.ylabel('AI IRQPE 313')
plt.title('Local precipitation for AI_IRQPE_313')

# 计算相关系数
correlation_coefficient = np.corrcoef(local_clean['QPESUMS'], local_clean['AI313'])[0, 1]
# 添加颜色条
cbar = plt.colorbar(sc)
cbar.set_label('Density')

# 在图上添加 RMSE 和相关系数
ax.annotate(f'RMSE: {RMSE:.2f}\ncorr: {correlation_coefficient:.2f}', 
            xy=(0.05, 0.95), 
            xycoords='axes fraction', 
            fontsize=12, 
            color='black',
            horizontalalignment='left', 
            verticalalignment='top',
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
# plt.savefig(os.path.join('C:/Users/haapy meow/Desktop/Topics/CWA/專題/AIQPE_code_2024/CC_pic/', "AI_IRQPE_313_QPESUM_all_flat_comparison.png"))  # 使用绝对路径
plt.show()
#################################################################################
# Dessity scatter plot 322小範圍
#################################################################################
# local_clean = local.replace([np.inf, -np.inf], np.nan).dropna()
fig, ax = plt.subplots(dpi=600, facecolor='white')
xy = np.vstack((local_clean['QPESUMS'], local_clean['AI322']));z = gaussian_kde(xy)(xy)
fig, ax = plt.subplots()
sc = ax.scatter(local_clean['QPESUMS'], local_clean['AI322'], c=z, s=1, cmap='viridis')
diff = np.subtract(local_clean['QPESUMS'], local_clean['AI322'])
square = np.square(diff);MSE = square.mean();RMSE = np.sqrt(MSE)
plt.plot(x, x, color='red')
ax.set_xscale('log')
ax.set_yscale('log')
# plt.xlim(min_val, 800);plt.ylim(, 800)
plt.xlabel('QPESUMS');plt.ylabel('AI IRQPE 322')
plt.title('Local precipitation for AI_IRQPE_322')

# 计算相关系数
correlation_coefficient = np.corrcoef(local_clean['QPESUMS'],local_clean['AI322'])[0, 1]
# 添加颜色条
cbar = plt.colorbar(sc)
cbar.set_label('Density')

# 在图上添加 RMSE 和相关系数
ax.annotate(f'RMSE: {RMSE:.2f}\ncorr: {correlation_coefficient:.2f}', 
            xy=(0.05, 0.95), 
            xycoords='axes fraction', 
            fontsize=12, 
            color='black',
            horizontalalignment='left', 
            verticalalignment='top',
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
# plt.savefig(os.path.join('C:/Users/haapy meow/Desktop/Topics/CWA/專題/AIQPE_code_2024/CC_pic/', "AI_IRQPE_313_QPESUM_all_flat_comparison.png"))  # 使用绝对路径
plt.show()



#%%
from scipy.stats import pearsonr

# 初始化存储空间相关性的数组
spatial_corr_AI313 = np.zeros_like(QPESUMS[0])
spatial_corr_AI322 = np.zeros_like(QPESUMS[0])

# 遍历每个网格点，计算空间相关性
for lat in range(QPESUMS.shape[1]):
    for lon in range(QPESUMS.shape[2]):
        # 提取时间序列
        series_QPESUMS = QPESUMS[indices[0]:indices[-1], lat, lon]
        series_AI313 = AI313[indices[0]:indices[-1], lat, lon]
        series_AI322 = AI322[indices[0]:indices[-1], lat, lon]
        
        # 计算AI313与QPESUMS的相关系数
        if not np.isnan(series_QPESUMS).all() and not np.isnan(series_AI313).all():
            spatial_corr_AI313[lat, lon], _ = pearsonr(series_QPESUMS, series_AI313)
        else:
            spatial_corr_AI313[lat, lon] = np.nan

        # 计算AI322与QPESUMS的相关系数
        if not np.isnan(series_QPESUMS).all() and not np.isnan(series_AI322).all():
            spatial_corr_AI322[lat, lon], _ = pearsonr(series_QPESUMS, series_AI322)
        else:
            spatial_corr_AI322[lat, lon] = np.nan

# 绘制相关系数图
fig, axs = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True, facecolor='white')
fig.subplots_adjust(wspace=0.05)  # 减小子图之间的宽度间距
m = Basemap(projection='cyl', resolution='i', fix_aspect=True,
            llcrnrlon=120, llcrnrlat=22, urcrnrlon=122.5, urcrnrlat=25.5, lat_ts=22)

# 绘制AI313相关系数图
ax = axs[0]
cx, cy = np.meshgrid(theLons, theLats)
c = m.contourf(cx, cy, spatial_corr_AI313, np.linspace(0, 1, 21), cmap='rainbow', norm=mpl.colors.Normalize(vmin=0, vmax=1), ax=ax)
m.drawparallels(np.arange(22, 25.5, 2), labels=[1, 0, 0, 0], color='#787878', fontsize=12, ax=ax)
m.drawmeridians(np.arange(120, 122.5, 2), labels=[0, 1, 0, 1], color='#787878', fontsize=12, ax=ax)
m.drawcoastlines(color='#28FF28', ax=ax)
ax.set_title("Spatial Correlation AI313 vs QPESUMS", fontsize=12)

# 绘制AI322相关系数图
ax = axs[1]
c = m.contourf(cx, cy, spatial_corr_AI322, np.linspace(0, 1, 21), cmap='rainbow', norm=mpl.colors.Normalize(vmin=0, vmax=1), ax=ax)
m.drawparallels(np.arange(22, 25.5, 2), labels=[1, 0, 0, 0], color='#787878', fontsize=12, ax=ax)
m.drawmeridians(np.arange(120, 122.5, 2), labels=[0, 1, 0, 1], color='#787878', fontsize=12, ax=ax)
m.drawcoastlines(color='#28FF28', ax=ax)
ax.set_title("Spatial Correlation AI322 vs QPESUMS", fontsize=12)

# 共用 colorbar
fig.colorbar(c, ax=axs, orientation='vertical', fraction=.02, pad=0.04).set_label('Correlation Coefficient')

# 保存和显示图像
valid_filename = re.sub(r'[^\w\-_\. ]', '_', "spatial_correlation.png")
# plt.savefig(os.path.join('C:/Users/haapy meow/Desktop/Topics/CWA/專題/AIQPE_code_2024/CC_pic/', valid_filename))  # 使用绝对路径
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
f, ax = plt.subplots(figsize=(4, 6))

# 绘制小提琴图
sns.violinplot(x='Group', y='Coefficient', data=data, bw_adjust=.5, cut=1, linewidth=1, palette="Set3")


# 设置 y 轴范围
ax.set(ylim=(-.7, 1.05))
ax.set_ylim(-0.2, 1)
ax.set_yticks(np.arange(-0.3, 1.1, 0.1))  # 从-0.2到1.0，每0.1一横
ax.set_yticklabels([f'{tick:.1f}' for tick in np.arange(-0.3, 1.1, 0.1)])  # 设置y轴刻度标签


# 去除边框
sns.despine(left=True, bottom=True)
plt.title('CC of AI_IRPQE_313 vs AI_IRQPE_322')

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
f, ax = plt.subplots(figsize=(4, 6))

# 绘制小提琴图
sns.violinplot(x='Group', y='Bias', data=data, bw_adjust=.5, cut=1, linewidth=1, palette="Set3")


# 设置 y 轴范围
ax.set(ylim=(-.7, 1.05))
ax.set_ylim(-30, 30)
ax.set_yticks(np.arange(-30, 30, 5))  # 从-0.2到1.0，每0.1一横
ax.set_yticklabels([f'{tick:.1f}' for tick in np.arange(-30, 30, 5)])  # 设置y轴刻度标签



# 去除边框
sns.despine(left=True, bottom=True)
plt.title('Bias of AI_IRPQE_313 vs AI_IRQPE_322')

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
