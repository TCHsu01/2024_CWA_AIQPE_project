import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, cm
import matplotlib as mpl
import os
import datetime
from scipy import interpolate
from scipy.interpolate import interp2d, RectBivariateSpline
# 進行資料讀取
direct = ['H:/2024cwa/AI_IRQPE_322/']
file = sorted(os.listdir('H:/2024cwa/AI_IRQPE_322/'))  # 對檔案進行排序，以確保正確的次序

cc = 0
dd = 0
hourly_qpe = np.zeros((len(file), 248, 200))
length = len(file)
hour_remind = np.zeros(length)
day_remind = np.zeros(length)
min_remind = np.zeros(length)  # 儲存分鐘資訊
# current_date = datetime.datetime(year=2023, month=6, day=22)  # 這裡的年份2023是一個假設，你可能需要調整它
# day_labels = []
for a in range(len(direct)):
    for b in range(len(file)):
        with open(direct[a] + file[b], "rb") as fid:
            test = np.fromfile(fid, dtype='float32')
            test = np.reshape(test, (248, 200))
            test = test[::-1, :]
            test[test <= 0] = np.nan
            filename = file[b]
            date_str = filename.split('_')[-1]
            tt11 = date_str.split('.')[0]
            # day_labels.append(tt11[13:21])  # 根據你的檔名格式修改
            if int(tt11[len(tt11)-1:len(tt11)]) == 10:
                hourly_qpe[dd] = test
                hour_remind[dd] = tt11[8:9]
                min_remind[dd] = tt11[10:11]  # 儲存分鐘資訊
                day_remind[dd] = tt11[6:7]
                dd += 1
                # day_labels.append(current_date.strftime('%Y%m%d'))
                # current_date += datetime.timedelta(days=1)  # 遞增一天
# 將雨量資料組織成 {日期: {小時: {分鐘: 雨量資料}}}
organized_data = {}
for idx in range(len(day_remind)):
    day = int(day_remind[idx])
    hour = int(hour_remind[idx])
    minute = int(min_remind[idx])
    if day not in organized_data:
        organized_data[day] = {}
    if hour not in organized_data[day]:
        organized_data[day][hour] = {}
    organized_data[day][hour][minute] = hourly_qpe[idx]

# 根據organized_data計算每天的總雨量
num_days = len(organized_data)
rainfall_3d = np.zeros((num_days, 248, 200))

day_index = 0


for day, hours_data in organized_data.items():
    total_rainfall = np.zeros((248, 200))
    cc=0
    for hour in range(0, 24):  # 從0到23小時
        
        # 優先使用10分的資料
        if 10 in hours_data.get(hour, {}):
            total_rainfall = np.nansum([total_rainfall, hours_data[hour][10]], axis=0)
            cc+=1
    print(cc)
    
    if cc < 18:
        total_rainfall[:] = np.nan
        # 如果10分的資料不存在，嘗試使用前後10分鐘的資料填充
        # else:
        #     prev_minute = 0
        #     next_minute = 20
            
        #     if prev_minute in hours_data.get(hour, {}):
        #         total_rainfall = np.nansum([total_rainfall, hours_data[hour][prev_minute]], axis=0)
        #         print('with00')
        #     elif next_minute in hours_data.get(hour, {}):
        #         total_rainfall = np.nansum([total_rainfall, hours_data[hour][next_minute]], axis=0)
        #         print('with20')
        #     else:
        #         # 如果0分和20分的資料都缺失，使用np.nan填充整天的資料
        #         total_rainfall[:] = np.nan
        #         break

    rainfall_3d[day_index] = total_rainfall
    day_index += 1
# 現在daily_rainfall字典中包含每天的總雨量資訊




sorted_days = sorted(organized_data.keys())
# for day in range(len(rainfall_3d)):
#     plt.figure(dpi=600)
    
#     theLons = np.arange(119,122.98,0.02)
#     theLats = np.arange(21.02,25.96,0.02)
#     m = Basemap(projection='cyl', resolution='i', fix_aspect=True, llcrnrlon=118, llcrnrlat=19.99, urcrnrlon=123.5, urcrnrlat=26.99, lat_ts=20)
#     cx, cy = np.float32(np.meshgrid(theLons, theLats))
#     c = m.contourf(cx, cy, rainfall_3d[day], clevs,shading='gouraud', cmap=cmap,norm=norm)
    
#     y_axis = m.drawparallels(np.arange(21, 26, 2), labels=[1, 0, 0, 0], color='#787878', fontsize=10)
#     x_axis = m.drawmeridians(np.arange(119, 123.01, 2), labels=[0, 1, 0, 1], color='#787878', fontsize=10)
#     m.drawcoastlines(color='#28FF28')
#     plt.colorbar(c).set_label('Rainfall rate(mm/hr)')
#     # plt.title(day_labels[day])
#     plt.title(str(sorted_days[day]))

theLons = np.arange(119,122.98,0.02)
theLats = np.arange(21.02,25.96,0.02)
cx, cy = np.float32(np.meshgrid(theLons, theLats))

RR_01 = np.zeros((len(rainfall_3d),25,35))
m = Basemap(projection='cyl', resolution='i', fix_aspect=True, llcrnrlon=118, llcrnrlat=19.99, urcrnrlon=123.5, urcrnrlat=26.99, lat_ts=20)
for day in range(len(rainfall_3d)):    
    fx = interpolate.interp2d(theLats, theLons, cx.T, kind='linear')  #網格點也得一併內插
    fy = interpolate.interp2d(theLats, theLons, cy.T, kind='linear')
    fz = interpolate.interp2d(theLats, theLons, rainfall_3d[day].T, kind='linear')          
    xlon = np.arange(119,123.01,0.1)    #110格      
    ylon = np.arange(21.02,25.96,0.1)      #140格
    cxnew = fx(ylon, xlon) 
    cynew = fy(ylon, xlon)     
    cznew = fz(ylon, xlon)
    
    cznew[cznew<0] = np.nan
    
    

    cxnew = cxnew[10:35,10:45]
    cynew = cynew[10:35,10:45]
    cznew = cznew[10:35,10:45]
    for j in range(25):
        for k in range(35):
            lon, lat = m(cxnew[j, k], cynew[j, k])
            is_land = m.is_land(lon, lat)
            if not is_land:
                cznew[ j, k] = np.nan
                
    RR_01[day] = cznew
    
    # np.save('AI_IRQPE_2023'+str(sorted_days[day])+'.npy',RR_01[day])
    
# RR_01 = np.asarray(RR_01,dtype='float32')
np.save('AI_IRQPE.npy',RR_01)
np.save('sorted_days.npy',sorted_days)

for day in range(len(rainfall_3d)):      
    filename = 'AI_IRQPE_2023' + str(sorted_days[day]) + '.bin'
    print(RR_01.shape)
    RR_01[day].flatten()
    RR_01[day].tofile(filename)
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

clevs, cmaplist = setcolorQPF(cnum=3)
cmap = (mpl.colors.ListedColormap(cmaplist/255.).with_extremes(
        over=(253/255., 201/255., 255/255.), under='white'))
norm = mpl.colors.BoundaryNorm(clevs, cmap.N)
clevs_labels = [str(v) if isinstance(v, float) else str(int(v)) for v in clevs]

# for day in range(len(rainfall_3d)):  
#     plt.figure(dpi=600)
#     m = Basemap(projection='cyl', resolution='i', fix_aspect=True, llcrnrlon=120, llcrnrlat=21.5, urcrnrlon=122, urcrnrlat=25.5, lat_ts=20)
#     cx, cy = np.float32(np.meshgrid(theLons, theLats))
#     c = m.contourf(cxnew, cynew, RR_01[day], clevs,shading='gouraud', cmap=cmap,norm=norm)
    
#     y_axis = m.drawparallels(np.arange(21.5, 25.51, 2), labels=[1, 0, 0, 0], color='#787878', fontsize=10)
#     x_axis = m.drawmeridians(np.arange(120, 122.01, 2), labels=[0, 1, 0, 1], color='#787878', fontsize=10)
#     m.drawcoastlines(color='#28FF28')
#     plt.colorbar(c).set_label('Rainfall rate(mm/hr)')
#     plt.title(str(sorted_days[day]))

from netCDF4 import Dataset

# Assume RR_01 is your data array with shape (num_days, 25, 35)

# Create a new netCDF file
ncfile = Dataset('AI_IRQPE.nc', 'w', format='NETCDF4_CLASSIC')

# Define the dimensions
day_dim = ncfile.createDimension('day', len(RR_01))  # assuming RR_01 has shape (num_days, 25, 35)
lat_dim = ncfile.createDimension('lat', 25)
lon_dim = ncfile.createDimension('lon', 35)

# Create variables
days_var = ncfile.createVariable('day', np.int32, ('day',))
lats_var = ncfile.createVariable('latitude', np.float32, ('lat',))
lons_var = ncfile.createVariable('longitude', np.float32, ('lon',))
rain_var = ncfile.createVariable('rainfall', np.float32, ('day', 'lat', 'lon',))

# Assuming you have arrays for the days, latitudes, and longitudes
days_var[:] = sorted_days  # or your array of days
lats_var[:] = np.arange(120,122.41,0.1)    # your latitude array that matches the lat_dim size
lons_var[:] = np.arange(22.02,25.42,0.1)  # your longitude array that matches the lon_dim size

# Write the data to the variable for each day
for day in range(len(RR_01)):
    rain_var[day, :, :] = RR_01[day]

# Close the file
ncfile.close()
