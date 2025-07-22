import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import matplotlib as mpl
import os
from datetime import datetime

# set path
output_header = input('313 or 322 :')
if output_header=='313':
    direct = ['H:/2024cwa/AI_IRQPE_313_rerun/']#['H:/2024cwa/GAEMI/AI_313/']
else:
    direct = ['H:/2024cwa/GAEMI/AI_322/']
print(f"direct : {direct[0]}")
file = os.listdir(direct[0])

# 時間範圍
start_date = datetime.strptime('2024-05-01 00:00', '%Y-%m-%d %H:%M')
end_date = datetime.strptime('2024-06-30 23:00', '%Y-%m-%d %H:%M')
print(f"time range : {start_date} to {end_date}")

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
cmap = (mpl.colors.ListedColormap(cmaplist/255.).with_extremes(
    over=(253/255., 201/255., 255/255.), under='white'))
norm = mpl.colors.BoundaryNorm(clevs, cmap.N)
clevs_labels = [str(v) if isinstance(v, float) else str(int(v)) for v in clevs]

# 解析文件名中的日期
def get_date_from_filename(filename):
    date_str = filename.split('_')[-1].split('.')[0]
    return datetime.strptime(date_str, "%Y%m%d%H%M")

# 获取文件中的日期
file_dates = [get_date_from_filename(f) for f in file]

# 筛选整点的文件
hourly_files = [file[i] for i, date in enumerate(file_dates) if date.minute == 0]
hourly_dates = [date for date in file_dates if date.minute == 0]

# 按日期分组文件
date_groups = {}
for idx, date in enumerate(hourly_dates):
    date_str = date.strftime("%Y-%m-%d")
    if start_date <= date <= end_date:
        if date_str not in date_groups:
            date_groups[date_str] = []
        date_groups[date_str].append(hourly_files[idx])
# 資料缺值紀錄
daily_aiqpe_list = []
date_list = []
missing_files_summary = {}
missing_hours_summary = {}

# daily processing
for date_str, files in date_groups.items():
    formatted_date = datetime.strptime(date_str, '%Y-%m-%d')
    hourly_qpe = np.full((24, 248, 200), np.nan)
    missing_hours = []
    missing_files = set()
    m = Basemap(projection='cyl', resolution='i', fix_aspect=True, llcrnrlon=118, llcrnrlat=19.99, urcrnrlon=123.5, urcrnrlat=26.99, lat_ts=20)

    # 檢查每小時資料有效性
    for hour in range(24):
        expected_file = f"model_hourly_{formatted_date.strftime('%Y%m%d')}{hour:02d}00.bin"
        if expected_file not in files:
            missing_files.add(expected_file)
            continue

        if expected_file in files:
            file_idx = files.index(expected_file)
            print(f"Processing file: {expected_file}")
            with open(direct[0] + expected_file, "rb") as fid:
                test = np.fromfile(fid, dtype='float32')
                test = np.reshape(test, (248, 200))
                test = test[::-1, :]
                test[test <= 0] = np.nan

                theLons = np.arange(119, 122.98, 0.02)
                theLats = np.arange(21.02, 25.96, 0.02)
                # theLons = np.arange(118,123.49,0.1)
                # theLats = np.arange(20,26.99,0.1)

                if np.all(np.isnan(test)):
                    print(f"File {expected_file} has all NaN values. Skipping...")
                    missing_hours.append(f"{formatted_date.strftime('%Y-%m-%d')} {hour:02d}:00")
                else:
                    hourly_qpe[hour] = test

    # 記錄缺資料檔案名稱
    if missing_files:
        missing_files_summary[date_str] = list(missing_files)
    
    # 同上，但記錄小時
    if missing_hours:
        missing_hours_summary[date_str] = missing_hours

    # 記錄全都是 NaN 的資料
    if np.all(np.isnan(hourly_qpe)):
        print(f"All data for {date_str} are NaN. Skipping...")
        continue

    
    ###### make data list ######
    daily_qpe = np.nansum(hourly_qpe, axis=0)
    daily_aiqpe_list.append(daily_qpe)

    # plot daily QPE
    plt.figure(dpi=600, facecolor='white')
    m = Basemap(projection='cyl', resolution='i', fix_aspect=True, llcrnrlon=118, llcrnrlat=19.99, urcrnrlon=123.5, urcrnrlat=26.99, lat_ts=20)
    cx, cy = np.float32(np.meshgrid(theLons, theLats))
    c = m.contourf(cx, cy, daily_qpe, clevs, cmap=cmap, norm=norm)

    y_axis = m.drawparallels(np.arange(21, 26, 2), labels=[1, 0, 0, 0], color='#787878', fontsize=10)
    x_axis = m.drawmeridians(np.arange(119, 123.01, 2), labels=[0, 1, 0, 1], color='#787878', fontsize=10)
    m.drawcoastlines(color='#28FF28')
    plt.colorbar(c).set_label('Rainfall rate(mm/hr)')
    plt.title(f"{output_header} Daily QPE for {formatted_date.strftime('%Y-%m-%d')}", fontsize=12)
    plt.annotate(f"nan: {len(missing_hours)}/{24-len(list(missing_files))}", xy=(0.972,0.937), xycoords='axes fraction', fontsize=10, 
            horizontalalignment='right', verticalalignment='bottom',bbox=dict(facecolor='white', alpha=0.8))
    date_list.append(formatted_date.strftime('%Y-%m-%d'))
    plt.savefig(f"daily_ai_irqpe_{date_str}.png", bbox_inches='tight')
    plt.show()
        



# 輸出分別為"缺資料" 與 "全nan" 的檔案
with open("missing_files_summary.txt", "w") as f:
    for date, files in missing_files_summary.items():
        f.write(f"Date: {date}\n")
        f.write(f"Missing files: {', '.join(files)}\n\n")

with open("missing_hours_summary.txt", "w") as f:
    for date, hours in missing_hours_summary.items():
        f.write(f"Date: {date}\n")
        f.write(f"Missing data hours: {', '.join(hours)}\n\n")



# export daily_aiqpe_list as .npy
qpe_output_fn = f"{output_header}_daily_aiqpe_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.npy"
date_output_fn = f"date_list_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.npy"
np.save(qpe_output_fn, daily_aiqpe_list)
np.save(date_output_fn, date_list)
