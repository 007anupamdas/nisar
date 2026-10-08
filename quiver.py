import csv
from datetime import datetime as dt
import pandas as pd
import numpy as np
# from osgeo import ogr
import pyproj as pp
from pyproj import CRS, Transformer
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import warnings
import os
from PIL import Image

# import cv2
import matplotlib.image as mpim

warnings.filterwarnings("ignore")

#xl_path = r'F:\anup\novasar\27Aug2021_Sce18_Kanchanjanga_IIRS.csv'


def wgs_to_utm(lon, lat):
    zone = int(np.floor(((180 + lon) / 6) % 60) + 1)
    is_northern = 'south' if lat < 0 else 'north'

    to_crs = CRS.from_proj4("+proj=utm +zone="+str(zone)+' +'+is_northern+' +ellps=WGS84 +datum=WGS84 +units=m +no_defs')
    from_crs = CRS.from_epsg(4326)

    proj = Transformer.from_crs(from_crs, to_crs, always_xy=True)
    return proj.transform(lon, lat)


# PATH = r'D:\eos-06\newtilt25\kotta'
# for fil in os.listdir(PATH):
#     if fil.endswith('.jpg'):
#         fi = fil.split('.jpg')[0]
xl_path = r'D:\a\pts.csv'
img_path = r'D:\a\img.jpg'
img = mpim.imread(img_path)
met_path = r'D:\a\meta.met'
with open(met_path) as f:
    meta_list = f.readlines()
    metalist1 = [x.split('\n') for x in meta_list]
    metalist = [x[0].split(':') for x in metalist1]
    for i in range(0, len(metalist) - 1):
        if 'ProdULLat' in metalist[i][0]:
            ULLat = float(metalist[i][1].rstrip(','))
        elif 'ProdULLon' in metalist[i][0]:
            ULLon = float(metalist[i][1].rstrip(','))
            # ULLon = 180 - ULLon1 if ULLon1 < 0 else ULLon1
        elif 'ProdURLat' in metalist[i][0]:
            URLat = float(metalist[i][1].rstrip(','))
        elif 'ProdURLon' in metalist[i][0]:
            URLon = float(metalist[i][1].rstrip(','))
            # URLon = 180 - URLon1 if URLon1 < 0 else URLon1
        elif 'ProdLLLat' in metalist[i][0]:
            LLLat = float(metalist[i][1].rstrip(','))
        elif 'ProdLLLon' in metalist[i][0]:
            LLLon = float(metalist[i][1].rstrip(','))
            # LLLon = 180 - LLLon1 if LLLon1 < 0 else LLLon1
        elif 'ProdLRLat' in metalist[i][0]:
            LRLat = float(metalist[i][1].rstrip(','))
        elif 'ProdLRLon' in metalist[i][0]:
            LRLon = float(metalist[i][1].rstrip(','))
            # LRLon = 180 - LRLon1 if LRLon1 < 0 else LRLon1
        elif 'NoScans' in metalist[i][0]:
            height = int(metalist[i][1].rstrip(','))
        elif 'NoPixels' in metalist[i][0]:
            width = int(metalist[i][1].rstrip(','))
        elif 'ProdULMapX' in metalist[i][0]:
            ULX = float(metalist[i][1].rstrip(','))
            # ULLon = 180 - ULLon1 if ULLon1 < 0 else ULLon1
        elif 'ProdULMapY' in metalist[i][0]:
            ULY = float(metalist[i][1].rstrip(','))
        elif 'ProdURMapX' in metalist[i][0]:
            URX = float(metalist[i][1].rstrip(','))
        elif 'ProdURMapY' in metalist[i][0]:
            URY = float(metalist[i][1].rstrip(','))
            # URLon = 180 - URLon1 if URLon1 < 0 else URLon1
        elif 'ProdLLMapY' in metalist[i][0]:
            LLY = float(metalist[i][1].rstrip(','))
        elif 'ProdLLMapX' in metalist[i][0]:
            LLX = float(metalist[i][1].rstrip(','))
            # LLLon = 180 - LLLon1 if LLLon1 < 0 else LLLon1
        elif 'ProdLRMapY' in metalist[i][0]:
            LRY = float(metalist[i][1].rstrip(','))
        elif 'ProdLRMapX' in metalist[i][0]:
            LRX = float(metalist[i][1].rstrip(','))
        elif 'StripID' in metalist[i][0]:
            strip = int(metalist[i][1].rstrip(','))
        elif 'SceneSequenceNo' in metalist[i][0]:
            scene = int(metalist[i][1].rstrip(','))
        elif 'Path' in metalist[i][0]:
            pth = metalist[i][1].rstrip(',')
        elif 'TiltAngle' in metalist[i][0]:
            tilt = float(metalist[i][1]).rstrip(',')
        elif 'Row' in metalist[i][0]:
            rw = metalist[i][1].rstrip(',')
        elif 'AcquistionMode' in metalist[i][0]:
            mode = metalist[i][1]
        elif 'DateOfPass' in metalist[i][0]:
            dop = dt.strptime(metalist[i][1].rstrip(',').strip(), '"%d-%b-%Y"')
            dop1 = metalist[i][1].strip()

# xl = pd.read_csv(xl_path)
with open(xl_path,'r') as xl:
    cvs=csv.reader(xl,delimiter=',')

    lat_inp1_ = []
    lon_inp1_ = []
    x_inp1 = []
    y_inp1 = []
    x_ref = []
    y_ref = []
    # lat_inp2_ = []
    # lon_inp2_ = []
    # lat_inp3_ = []
    # lon_inp3_ = []
    lat_ref_ = []
    lon_ref_ = []
    # lat_err_inp1_ = []
    # lon_err_inp1_ = []
    # lat_err_inp2_ = []
    # lon_err_inp2_ = []
    # lat_err_inp3_ = []
    # lon_err_inp3_ = []

    for col in cvs:
        lat_inp1_.append(col[7])
        lon_inp1_.append(col[8])
        y_inp1.append(col[5])
        x_inp1.append(col[6])
        # lat_inp2_.append(col[4])
        # lon_inp2_.append(col[3])
        # lat_inp3_.append(col[7])
        # lon_inp3_.append(col[6])
        lat_ref_.append(col[13])
        lon_ref_.append(col[14])
        x_ref.append(col[12])
        y_ref.append(col[11])
        # lat_err_inp1_.append(col[4])
        # lon_err_inp1_.append(col[5])
        # lat_err_inp2_.append(col[16])
        # lon_err_inp2_.append(col[15])
        # lat_err_inp3_.append(col[19])
        # lon_err_inp3_.append(col[18])

# ULX, ULY = wgs_to_utm(ULLon, ULLat)
# LLX, LLY = wgs_to_utm(LLLon, LLLat)
# URX, URY = wgs_to_utm(URLon, URLat)
# LRX, LRY = wgs_to_utm(LRLon, LRLat)

lat_inp1 = lat_inp1_
# lat_inp2 = lat_inp2_[2:]
# lat_inp3 = lat_inp3_[4:]
lon_inp1 = lon_inp1_
# lon_inp2 = lon_inp2_[4:]
# lon_inp3 = lon_inp3_[4:]
lat_ref = lat_ref_
lon_ref = lon_ref_


# lat_err_inp2 = lat_err_inp2_[4:]
# lat_err_inp3 = lat_err_inp3_[4:]
# lon_err_inp1 = lon_err_inp1_[2:]
# lon_err_inp2 = lon_err_inp2_[4:]
# lon_err_inp3 = lon_err_inp3_[4:]

lat_inp1 = [float(item) for item in lat_inp1]
# lat_inp2 = [float(item) for item in lat_inp2]
# lat_inp3 = [float(item) for item in lat_inp3]
lon_inp1 = [float(item) for item in lon_inp1]
# lon_inp2 = [float(item) for item in lon_inp2]
# lon_inp3 = [float(item) for item in lon_inp3]
lat_ref = [float(item) for item in lat_ref]
lon_ref = [float(item) for item in lon_ref]

x_inp1 = [float(item) for item in x_inp1]
y_inp1 = [float(item) for item in y_inp1]

x_ref = [float(item) for item in x_ref]
y_ref = [float(item) for item in y_ref]

lat_err_inp1 = [lat_inp1[i]- lat_ref[i] for i in range(0, len(lat_inp1))]
lon_err_inp1 = [lon_inp1[i]- lon_ref[i] for i in range(0, len(lat_inp1))]

x_err = [x_ref[i]- x_inp1[i] for i in range(0, len(lat_inp1))]
y_err = [y_ref[i]- y_inp1[i] for i in range(0, len(lat_inp1))]

alo_dir = 'N' if np.mean(x_err) > 0 else 'S'
acs_dir = 'E' if np.mean(x_err) > 0 else 'W'

# lat_err_inp1 = [float(item) for item in lat_err_inp1]
# lat_err_inp2 = [float(item) for item in lat_err_inp2]
# lat_err_inp3 = [float(item) for item in lat_err_inp3]
# lon_err_inp1 = [float(item) for item in lon_err_inp1]
# lon_err_inp2 = [float(item) for item in lon_err_inp2]
# lon_err_inp3 = [float(item) for item in lon_err_inp3]


min_lon = lon_inp1[np.argmin(lon_inp1)]
max_lon = lon_inp1[np.argmax(lon_inp1)]
min_lat = lat_inp1[np.argmin(lat_inp1)]
max_lat = lat_inp1[np.argmax(lat_inp1)]

# inp = [list(pp.transform(crs_from,crs_to,lon_inp[i], lat_inp[i])) for i in range(0,np.shape(lat_inp)[0])]
# ref = [list(pp.transform(crs_from,crs_to,lon_ref[i], lat_ref[i])) for i in range(0,np.shape(lat_inp)[0])]
#
# x_inp = [inp[i][0] for i in range(0, np.shape(inp)[0])]
# y_inp = [inp[i][1] for i in range(0, np.shape(inp)[0])]
# x_ref = [ref[i][0] for i in range(0, np.shape(inp)[0])]
# y_ref = [ref[i][1] for i in range(0, np.shape(inp)[0])]
#
lat_err = [(lat_ref[i]-lat_inp1[i]) for i in range(0,np.shape(lat_inp1)[0])]
lon_err = [(lon_ref[i]-lon_inp1[i]) for i in range(0,np.shape(lon_inp1)[0])]
#
laterr_ = [lat_err[i]*111*1000*np.cos(ULLat) for i in range(0,np.shape(lat_inp1)[0])]
lonerr_ = [lon_err[i]*111*1000*np.cos(ULLat) for i in range(0,np.shape(lon_inp1)[0])]
#
lat_sq = [np.square(item) for item in laterr_]
alo_rms = np.sqrt(np.sum(lat_sq)/len(lat_sq))
lon_sq = [np.square(item) for item in lonerr_]
acro_rms = np.sqrt(np.sum(lon_sq)/len(lon_sq))

x_sq = [np.square(item) for item in x_err]
alo_rms_map = np.sqrt(np.sum(x_sq)/len(lat_sq))
y_sq = [np.square(item) for item in y_err]
acro_rms_map = np.sqrt(np.sum(y_sq)/len(lon_sq))

##Internal distortion - Geo####
lat_lon_errcomb = [[lat_inp1[i],lon_inp1[i],lat_err_inp1[i],lon_err_inp1[i]] for i in range(0,np.shape(lat_inp1)[0])]
df = pd.DataFrame(lat_lon_errcomb)
df_sort = df.sort_values(by=[0,1], ascending=[False,True])

id_x = [np.absolute(df_sort[2][i]-df_sort[2][0]) for i in range(1,df[0].shape[0])]
id_y = [np.absolute(df_sort[3][i]-df_sort[3][0]) for i in range(1,df[0].shape[0])]

idx = round(np.sum(id_x)/(df[0].shape[0]-1),2)
idy = round(np.sum(id_y)/(df[0].shape[0]-1),2)

##Internal distortion - MAP####
map_errcomb = [[x_inp1[i],y_inp1[i],x_err[i],y_err[i]] for i in range(0,np.shape(lat_inp1)[0])]
dg = pd.DataFrame(map_errcomb)
dg_sort = df.sort_values(by=[0,1], ascending=[False,True])

id_x_map = [np.absolute(dg_sort[2][i]-dg_sort[2][0]) for i in range(1,df[0].shape[0])]
id_y_map = [np.absolute(dg_sort[3][i]-dg_sort[3][0]) for i in range(1,df[0].shape[0])]

idx_map = round(np.sum(id_x)/(df[0].shape[0]-1),4)
idy_map = round(np.sum(id_y)/(df[0].shape[0]-1),4)
ce90 = np.sqrt(np.square(acro_rms_map)+np.square(alo_rms_map))
#
# cum_lat = sum(laterr)
# cum_lon = sum(lonerr)
#
# alo_mean = np.sum(laterr)/len(laterr)
# acro_mean = np.sum(lonerr)/len(lonerr)
#
# if cum_lat > 0:
#     alo_dir = ' N'
# else:
#     alo_dir = ' S'
#
# if cum_lon > 0:
#     acro_dir = ' E'
# else:
#     acro_dir = ' W'
#
# along_track_error = str(round(alo_rms,2)) + alo_dir
# across_track_error = str(round(acro_rms,2)) + acro_dir
#
# CE90 = round(np.sqrt(np.sum(lat_sq+lon_sq)/len(lat_sq)),2)
#
#
# ce = np.sort([np.sqrt(lat_sq[i]+lon_sq[i]) for i in range(0,len(lat_sq))])
# ce90 = ce[round(0.9*len(ce)-1)]
#
#
# print("Along track error: {}".format(along_track_error))
# print("Across track error: {}".format(across_track_error))
# print("CE90: {}".format(round(ce90,2)))
# print("Internal Distortion : {} m, {} m".format(round(idx,2),round(idy,2)))
# # savefile = 'F:\\novasar\\IIRS_NRSC_STRIPMODE\\'+ xl_path.split('\\')[-1].split('.')[0]
# # savefile = 'F:\jayabharathi\irs_legacy\\'+ xl_path.split('\\')[-1].split('.')[0]
# savefile = 'F:\\anup\\novasar\\'+ xl_path.split('\\')[-1].split('.')[0]
# with open(savefile+'.txt','w') as f:
#     f.write("Along track error: " + str(along_track_error) + '\n')
#     f.write("Across track error: " + str(across_track_error) + '\n')
#     f.write("CE90: " + str(round(ce90,2)) + '\n')
#     f.write("Internal Distortion : " + str(round(idx,2)) + " m," + str(round(idy,2)) + " m")
#

fig, ax = plt.subplots(figsize=(38.4, 21.6))
# plt.imshow(img, extent=[ULLon, URLon, LLLat, ULLat], origin='upper')
plt.imshow(img, extent=[LLX,URX, LLY, URY], origin='upper')
# ax.quiver(lon_inp1,lat_inp1,lon_err,lat_err,color='y',headaxislength=5,headwidth=5,headlength=5,
#            minshaft=1,minlength=1,width=0.004, scale=1, angles='xy', scale_units='xy')

ax.quiver(x_inp1,y_inp1,x_err,y_err,color='magenta',width=0.002, scale=0.01, angles='xy', scale_units='xy', pivot='tail')

# ax.scatter(lon_inp1,lat_inp1,color='r')
# ax.quiver(lon_inp2,lat_inp2,lon_err_inp2,lat_err_inp2,color='g',headaxislength=2,headwidth=2,headlength=2,
#            minshaft=1,minlength=1,width=0.004, label='237419531')
# ax.quiver(lon_inp1,lat_inp3,lon_err_inp3,lat_err_inp3,color='b',headaxislength=2,headwidth=2,headlength=2,
#            minshaft=1,minlength=1,width=0.004, label='237419711')
plt.xlabel('Map Coordinates in Longitude direction', fontweight='bold', fontsize='18')
plt.ylabel('Map Coordinates in Latitude direction', fontweight='bold', fontsize='18')
plt.title("EOS-06 "+ mode+" Location Accuracy "+'\n'+"Date: "+dop1+ '\n'+ "Along Track Error: "+ str(round(alo_rms_map)) +' ' + alo_dir+
          "  Across Track Error: "+ str(round(acro_rms_map))+' ' +acs_dir+
          "  CE90: "+ str(round(ce90)), fontweight='bold', fontsize=18)
# plt.xticks([a for a in range(72,85)])
# plt.yticks([a for a in range(14,36)])
# plt.legend(loc='best')

plt.grid(linewidth=0.5)

ax.set_xticks(np.arange(LLX, URX,50000))
ax.set_yticks(np.arange(LLY, URY,50000))

# ax.xaxis.set_tick_params(labelsize=14, weight='bold')
# ax.yaxis.set_tick_params(labelsize=14, weight='bold')

ax.set_xticklabels([int(item) for item in ax.get_xticks()], weight='bold',size=16, rotation=90)
ax.set_yticklabels([int(item) for item in ax.get_yticks()], weight='bold', size=16, rotation=0)

# ax.set_xticklabels(np.arange(-800000, 800000,1000),fontsize=14)
# ax.set_yticklabels(np.arange(-900000, 900000,1000),fontsize=12)

# ax.set(xlim=(ULY, URY), ylim=(LLX, ULX))

plt.show()
plt.savefig('D:\\a\\out.png')
