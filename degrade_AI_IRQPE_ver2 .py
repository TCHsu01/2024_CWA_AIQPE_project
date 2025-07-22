#%%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import matplotlib as mpl
import os
from datetime import datetime

# set path
direct = 'H:/2024cwa/AIQPE_code_2024/'
file_list = os.listdir(direct)
opt1 = input('313 or 322 :')
opt2 = input('(1)230501~230831 (2)240501~240630 (3)Gaemi 240721~240728')
if opt2=='1':
    daily_aiqpe = np.load(os.path.join(direct, f'{opt1}_daily_aiqpe_20230501_20230831.npy'))
    date = np.load(os.path.join(direct, f'date_list_20230501_20230831.npy'))
    start_date = datetime.strptime('2023-05-01 00:00', '%Y-%m-%d %H:%M')
    end_date = datetime.strptime('2023-08-31 00:00', '%Y-%m-%d %H:%M')
elif opt2=='2':
    daily_aiqpe = np.load(os.path.join(direct, f'{opt1}_daily_aiqpe_20240501_20240630.npy'))
    date = np.load(os.path.join(direct, 'date_list_20240501_20240630.npy'))
    start_date = datetime.strptime('2024-05-01 00:00', '%Y-%m-%d %H:%M')
    end_date = datetime.strptime('2024-06-30 00:00', '%Y-%m-%d %H:%M')
elif opt2=='3':
    # print('not yet')
    daily_aiqpe = np.load(os.path.join(direct, f'{opt1}_daily_aiqpe_20240721_20240728.npy'))
    date = np.load(os.path.join(direct, 'date_list_20240721_20240728.npy'))
    start_date = datetime.strptime('2024-07-21 00:00', '%Y-%m-%d %H:%M')
    end_date = datetime.strptime('2024-07-28 23:50', '%Y-%m-%d %H:%M')


# daily_aiqpe = np.load(os.path.join(direct, '313_daily_aiqpe_20230507_20230507.npy'))
# date = np.load(os.path.join(direct, 'date_list_20230507_20230507.npy'))
degrade_aiqpe = []
test_aiqpe = []

# color setting
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
cmap = (mpl.colors.ListedColormap(cmaplist / 255.).with_extremes(
    over=(253 / 255., 201 / 255., 255 / 255.), under='white'))
norm = mpl.colors.BoundaryNorm(clevs, cmap.N)
clevs_labels = [str(v) if isinstance(v, float) else str(int(v)) for v in clevs]
#%%
# 降解析度
def shrink(data, rows, cols):
    row_sp = data.shape[0] // rows
    col_sp = data.shape[1] // cols
    tmp = np.sum(data[i::row_sp] for i in range(real_round((row_sp) // 5)))
    return np.sum(tmp[:, i::col_sp] for i in range(real_round((col_sp) // 5)))
def real_round(num, decimal=0):
    if decimal== 0:
        return int(num + 0.5)
    else:
        digit_value = 10 ** decimal
        return int(num * digit_value + 0.5) / digit_value


# daily processing
for i in range(len(date)):
    
    #原本AI_QPE的解析度
    theLons = np.arange(119, 122.98, 0.02)##
    theLats = np.arange(21.02, 25.96, 0.02)##

    # 先把目標區域圈出來（綠色虛線）
    tarLons = np.arange(120, 122.99, 0.02)##
    tarLats = np.arange(22, 25.99, 0.02)##

    #　目標區域再降解析度
    degrade_theLons = np.arange(120, 122.99, 0.1)##
    degrade_theLats = np.arange(22, 25.99, 0.1)##
    # 查找索引
    # AI origin range
    
    # m = Basemap(projection='cyl', resolution='i', fix_aspect=True, 
    # llcrnrlon=118, llcrnrlat=19.99, urcrnrlon=123.5, urcrnrlat=26.99, lat_ts=20)

    tarLons_indices = np.searchsorted(theLons, tarLons)##
    tarLats_indices = np.searchsorted(theLats, tarLats)##
    
    # degrade_theLons = np.arange(120, 122.5, 0.1)##
    # degrade_theLats = np.arange(22, 25.5, 0.1)##
    #####################################################
    formatted_date = date[i]                   # 日期
    tar_aiqpe = daily_aiqpe[i]                 # 原始資料
    var_aiqpe = tar_aiqpe[48:248,50:200]
    var_aiqpe = shrink(var_aiqpe, real_round(np.shape(var_aiqpe)[0] / 5), real_round(np.shape(var_aiqpe)[1] / 5))
    # var_aiqpe = var_aiqpe[1:36,:]

    fig, axs = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True, facecolor='white')
    m = Basemap(projection='cyl', resolution='i', fix_aspect=True,
                llcrnrlon=120, llcrnrlat=22, urcrnrlon=122.5, urcrnrlat=25.5, lat_ts=22)

    # 第一張：AI QPE
    ax = axs[0]
    cx, cy = np.meshgrid(theLons, theLats)
    c = m.contourf(cx, cy, daily_aiqpe[i], clevs, cmap=cmap, norm=norm, ax=ax)
    m.drawparallels(np.arange(22, 25.5, 2), labels=[1, 0, 0, 0], color='#787878', fontsize=12, ax=ax)
    m.drawmeridians(np.arange(120, 122.5, 2), labels=[0, 1, 0, 1], color='#787878', fontsize=12, ax=ax)
    m.drawcoastlines(color='#28FF28', ax=ax)
    ax.set_title(f"AI {opt1} Daily QPE for {date[i]}", fontsize=12)

    # 第二張：degrade
    ax = axs[1]
    cx, cy = np.meshgrid(degrade_theLons, degrade_theLats)
    c = m.contourf(cx, cy, var_aiqpe, clevs, cmap=cmap, norm=norm, ax=ax)#[tarLons_indices,tarLats_indices]
    m.drawparallels(np.arange(22, 25.5, 2), labels=[1, 0, 0, 0], color='#787878', fontsize=12, ax=ax)
    m.drawmeridians(np.arange(120, 122.5, 2), labels=[0, 1, 0, 1], color='#787878', fontsize=12, ax=ax)
    m.drawcoastlines(color='#28FF28', ax=ax)
    ax.set_title(f"{opt1} Degraded Daily QPE for {date[i]}", fontsize=12)

    # 第三張：island
    ax = axs[2]
    cx, cy = np.meshgrid(degrade_theLons, degrade_theLats)
    test = var_aiqpe.copy()
    for j in range(test.shape[0]):#np.shape(cx)[1]):#
        for k in range(test.shape[1]):#np.shape(cx)[0]):#
            lon, lat = cx[j, k], cy[j, k]
            is_land = m.is_land(lon, lat)
            if not is_land:
                test[j, k] = np.nan
            if lon < 120 and lat > 25.3:
                test[j, k] = np.nan
    c = m.contourf(cx, cy, test, clevs, cmap=cmap, norm=norm, ax=ax)
    m.drawparallels(np.arange(22, 25.5, 2), labels=[1, 0, 0, 0], color='#787878', fontsize=12, ax=ax)
    m.drawmeridians(np.arange(120, 122.5, 2), labels=[0, 1, 0, 1], color='#787878', fontsize=12, ax=ax)
    m.drawcoastlines(color='#28FF28', ax=ax)
    ax.set_title(f"{opt1} island degraded QPE for {date[i]}", fontsize=12)

    # 共用 colorbar
    fig.colorbar(c, ax=axs, orientation='vertical', fraction=.02, pad=0.04).set_label('Rainfall rate(mm/hr)')

    # 格式化日期并保存图像
    date_obj = datetime.strptime(date[i], "%Y-%m-%d")
    formatted_date = date_obj.strftime("%y%m%d")
    plt.savefig(os.path.join(direct, f"{opt1}_degrade_{formatted_date}.png"))  # 使用绝对路径
    plt.show()

    degrade_aiqpe.append(var_aiqpe)
    test_aiqpe.append(test)  # [20:55,20:45]

# degrade_output_fn = f"{opt1}_degrade_aiqpe_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.npy"
# island_output_fn = f"island_{opt1}_degrade_aiqpe_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.npy"


# np.save(os.path.join(direct, degrade_output_fn), degrade_aiqpe)
# np.save(os.path.join(direct, island_output_fn), test_aiqpe)

