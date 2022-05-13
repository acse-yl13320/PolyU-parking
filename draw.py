import os
import matplotlib.pyplot as plt
from tqdm import tqdm

path = 'meter_coord/'
dates = os.listdir(path)

for date in tqdm(dates[:]):
    files = [f for f in os.listdir(path + date) if f.endswith('csv')]
    # print(date)
    
    for file in tqdm(files[:]):
        # print(file)
        f = open(path + date + '/' + file, 'r')
        f.readline()
        lines = f.read().splitlines()
        f.close()
        
        plt.figure(figsize=(25, 15))
        bg = plt.imread('hk.png')
        plt.imshow(bg, extent=[113.84, 114.433, 22.194, 22.554])
        
        lat_O = []
        lon_O = []
        lat_V = []
        lon_V = []
        
        for line in lines[:]:
            x = line.split(',')
            try:
                id = x[0]
                lat = float(x[-2])
                lon = float(x[-1])
                if x[2] == 'O':
                    lat_O.append(lat)
                    lon_O.append(lon)
                else:
                    lat_V.append(lat)
                    lon_V.append(lon)        
            except Exception as e:
                # print(e)
                # print(line)
                pass
        
        plt.scatter(lon_O, lat_O, c='r', s=1, linewidths=None, alpha=0.2)
        plt.scatter(lon_V, lat_V, c='g', s=1, linewidths=None, alpha=0.2)
        plt.xlim(113.84, 114.433)
        plt.ylim(22.194, 22.554)
        # plt.text(113.9, 22.49, date + '/' + file[:-4])
        plt.title(date + '-' + file[0:2] + ':' + file[2:4] + ':' + file[4:6])
        plt.savefig(path + date + '/' + file[:-4] + '.png', pad_inches=0.)
        plt.close()
        
    