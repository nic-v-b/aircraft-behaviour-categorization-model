import math
import numpy as np
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import pandas as pd
import tqdm
import datetime
import time
from scipy.signal import resample
from sklearn.preprocessing import normalize, Normalizer, MinMaxScaler
# from tslearn.clustering import silhouette_score
from sklearn.metrics import silhouette_samples
from matplotlib.ticker import MaxNLocator
import plotly.express as px
px.set_mapbox_access_token('pk.eyJ1IjoibmljdmIiLCJhIjoiY2thNzBxMnl0MDAyYzJ0bmZpeW1jOHNlayJ9.p5h0jJ78qIUWcRLQ19muYw')

startTime = time.time()

fontsize_baseline = 12
# plt.style.use('ggplot')
# plt.rcParams['font.family'] = 'sans-serif'
# plt.rcParams['font.serif'] = 'Ubuntu'
# plt.rcParams['font.monospace'] = 'Ubuntu Mono'
# plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = fontsize_baseline
# plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = fontsize_baseline
plt.rcParams['xtick.labelsize'] = fontsize_baseline
plt.rcParams['ytick.labelsize'] = fontsize_baseline
# plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = fontsize_baseline
# plt.rcParams['image.cmap'] = 'jet'
# plt.rcParams['image.interpolation'] = 'none'
plt.rcParams['figure.figsize'] = (15, 10)
# plt.rcParams['axes.grid']=True
plt.rcParams['lines.linewidth'] = 0.25
# plt.rcParams['lines.markersize'] = 8
# colors = ['xkcd:pale orange', 'xkcd:sea blue', 'xkcd:pale red', 'xkcd:sage green', 'xkcd:terra cotta', 'xkcd:dull purple', 'xkcd:teal', 'xkcd: goldenrod', 'xkcd:cadet blue',
# 'xkcd:scarlet']

# set pandas dataframe display properties
max_rows = 25
max_cols = 15
# max_cols = None
pd.set_option('display.width', 320)
pd.set_option('display.max_columns', max_cols)
pd.set_option('display.min_rows', max_rows)
pd.set_option('display.max_rows', max_rows)

path = r'C:/Users/nicol/Google Drive/PhD/Conferences & Papers/AIAA Aviation 2023/Code and results/2_2. Aircraft behaviour category detection/'
# data_dir = r'C:/Users/nicol/Google Drive/PhD/Conferences & Papers/AIAA Aviation 2023/Code and results/2. Collect and process the selected data/'

# ac_info_data = pd.read_csv(path+'flight tracks/final data 1 day/smooth_tracks_dataset_all.csv')
ac_info_data = pd.read_csv(path+'flight tracks/final data 1 day/smooth_tracks_dataset.csv')
ac_info_data = ac_info_data.loc[:, ~ac_info_data.columns.str.contains('^Unnamed')]
print('ac_info_data = \n', ac_info_data)

# time_series_data = pd.read_csv(path+'temperature.csv')
# time_series_data = pd.read_csv(path + 'processed_dataset.csv')  # 24 h
time_series_data = pd.read_csv(path + 'smooth_tracks_processed_dataset.csv')  # 24 h
time_series_data_lon = pd.read_csv(path + 'flight tracks/final data 1 day/smooth_tracks_dataset_lon.csv')  # 24 h
time_series_data_lat = pd.read_csv(path + 'flight tracks/final data 1 day/smooth_tracks_dataset_lat.csv')  # 24 h
time_series_data = pd.read_csv(path + 'flight tracks/final data 1 day/smooth_tracks_dataset_alt.csv')  # 24 h
time_series_data_vel = pd.read_csv(path + 'flight tracks/final data 1 day/smooth_tracks_dataset_vel.csv')  # 24 h
time_series_data_hdg = pd.read_csv(path + 'flight tracks/final data 1 day/smooth_tracks_dataset_hdg.csv')  # 24 h
time_series_data_vrate = pd.read_csv(path + 'flight tracks/final data 1 day/smooth_tracks_dataset_vrate.csv')  # 24 h
time_series_data_accel = pd.read_csv(path + 'flight tracks/final data 1 day/smooth_tracks_dataset_accel.csv')  # 24 h
time_series_data_turnrate = pd.read_csv(path + 'flight tracks/final data 1 day/smooth_tracks_dataset_turnrate.csv')  # 24 h
# time_series_data = pd.read_csv(path + 'flight tracks/final data 2 days/smooth_tracks_dataset_alt.csv')  # 24 h
# time_series_data = pd.read_csv(path + 'processed_dataset_test.csv')  # 24 h
# time_series_data = time_series_data.sample(100) # for debugging
# time_series_data = time_series_data[0:50] # for debugging
# print('min test = ', min(time_series_data))
# print('min test = ', time_series_data.to_numpy().min())
# print('max test = ', time_series_data.to_numpy().max())
# print('min test = ', min(time_series_data.select_dtypes(include=[np.number]).min()))
# print('max test = ', max(time_series_data.select_dtypes(include=[np.number]).max()))
max_time_index = 150
time_series_data = time_series_data[0:max_time_index] # for debugging
time_series_data_lon = time_series_data_lon[0:max_time_index] # for debugging
time_series_data_lat = time_series_data_lat[0:max_time_index] # for debugging
time_series_data_vel = time_series_data_vel[0:max_time_index] # for debugging
time_series_data_hdg = time_series_data_hdg[0:max_time_index] # for debugging
time_series_data_vrate = time_series_data_vrate[0:max_time_index] # for debugging
time_series_data_accel = time_series_data_accel[0:max_time_index] # for debugging
time_series_data_turnrate = time_series_data_turnrate[0:max_time_index] # for debugging
# print('raw time_series_data =\n', time_series_data)
# print('raw time_series_data_lat =\n', time_series_data_lat)
# print('raw time_series_data_lon =\n', time_series_data_lon)
# print('raw time_series_data_hdg =\n', time_series_data_hdg)
time_series_data = time_series_data.loc[:, ~time_series_data.columns.str.contains('^Unnamed')]
time_series_data_lon = time_series_data_lon.loc[:, ~time_series_data_lon.columns.str.contains('^Unnamed')]
time_series_data_lat = time_series_data_lat.loc[:, ~time_series_data_lat.columns.str.contains('^Unnamed')]
time_series_data_vel = time_series_data_vel.loc[:, ~time_series_data_vel.columns.str.contains('^Unnamed')]
time_series_data_hdg = time_series_data_hdg.loc[:, ~time_series_data_hdg.columns.str.contains('^Unnamed')]
time_series_data_vrate = time_series_data_vrate.loc[:, ~time_series_data_vrate.columns.str.contains('^Unnamed')]
time_series_data_accel = time_series_data_accel.loc[:, ~time_series_data_accel.columns.str.contains('^Unnamed')]
time_series_data_turnrate = time_series_data_turnrate.loc[:, ~time_series_data_turnrate.columns.str.contains('^Unnamed')]


# # get df of all values from multiindex csv file
# time_series_data = pd.read_csv(path + 'flight tracks/final data 1 day/smooth_tracks_dataset_all.csv', header=[0, 1], index_col=0)  # 24 h
# # time_series_data = pd.read_csv(path + 'flight tracks/final data 2 days/smooth_tracks_dataset_all.csv', header=[0, 1], index_col=0)  # 24 h
# # time_series_data_all = time_series_data_all.loc[:, ~time_series_data_all.columns.str.contains('^Unnamed')]
# time_series_data = time_series_data[0:500]  # for debugging (limits the time intervals to first n
# print('raw time_series_data =\n', time_series_data)
#
# # get df of only alt values from multiindex csv file
# # time_series_data = time_series_data['lat']
# time_series_data = time_series_data['geoaltitude']
# print('raw time_series_data =\n', time_series_data)

# convert time to datetime objects
def unix_to_local(unix_time):
    local_time = datetime.datetime.fromtimestamp(unix_time)
    return local_time

def unix_to_utc(unix_time):
    # utc_time = datetime.datetime.utcfromtimestamp(unix_time).strftime('%Y-%m-%d %H:%M:%S')
    utc_time = datetime.datetime.utcfromtimestamp(unix_time)
    return utc_time

intervals = (
    ('weeks', 604800),  # 60 * 60 * 24 * 7
    ('days', 86400),    # 60 * 60 * 24
    ('hours', 3600),    # 60 * 60
    ('minutes', 60),
    ('seconds', 1),
)

def display_time(seconds, granularity=2):
    result = []

    for name, count in intervals:
        value = seconds // count
        if value:
            seconds -= value * count
            if value == 1:
                name = name.rstrip('s')
            result.append("{} {}".format(value, name))
    return ', '.join(result[:granularity])

# time_series_data['datetime'] = pd.to_datetime(time_series_data['datetime']) ------------------------------------- old version
# time_series_data['datetime'] = time_series_data['datetime'].apply(unix_to_local)


# cities_list = time_series_data.columns.tolist()[1:]  #all time_series_data column names except datetime column (list of cities)
# plt.plot(time_series_data['datetime'],time_series_data[cities_list[4]])
# # plt.show()

# # undersample_data = time_series_data.loc[np.linspace(time_series_data.index.min(),time_series_data.index.max(), 3000).astype(int)]  #resample 1500 points between min and max
# undersample_data = time_series_data.loc[0:100]  #resample 1500 points between min and max
# undersample_data = undersample_data.reset_index().drop('index', axis=1)
# time_series_data = undersample_data

# time_series_data = time_series_data.dropna(axis=1, how='all')
# time_series_data = time_series_data.dropna(axis=1, how='all', inplace=True)
# time_series_data.fillna(method='bfill', inplace=True)
time_series_data.fillna(method='ffill', inplace=True)
time_series_data_lon.fillna(method='ffill', inplace=True)
time_series_data_lat.fillna(method='ffill', inplace=True)
time_series_data_vel.fillna(method='ffill', inplace=True)
time_series_data_hdg.fillna(method='ffill', inplace=True)
time_series_data_vrate.fillna(method='ffill', inplace=True)
time_series_data_accel.fillna(method='ffill', inplace=True)
time_series_data_turnrate.fillna(method='ffill', inplace=True)
# filler = 0
# # filler = 999
# time_series_data.fillna(filler, inplace=True)
# time_series_data_lon.fillna(filler, inplace=True)
# time_series_data_lat.fillna(filler, inplace=True)
# time_series_data_vel.fillna(filler, inplace=True)
# time_series_data_hdg.fillna(filler, inplace=True)
# time_series_data_vrate.fillna(filler, inplace=True)
# time_series_data_accel.fillna(filler, inplace=True)
# time_series_data_turnrate.fillna(filler, inplace=True)
# print('time_series_data 1 =\n', time_series_data)


# time_series_data = normalize(time_series_data)
# time_series_data = (time_series_data-time_series_data.mean())/time_series_data.std()
# time_series_data = (time_series_data-time_series_data.min())/(time_series_data.max()-time_series_data.min())
# print('time_series_data 2 =\n', time_series_data)


cols_list = list(time_series_data.columns)
# print('cols = ', cols_list)

# # for debugging
# # used to limit the number of diff tracks to run faster on limited number of tracks
# lim_time_idx = 100
# time_series_data = time_series_data.iloc[:, 0:lim_time_idx]
# time_series_data_lon = time_series_data_lon.iloc[:, 0:lim_time_idx]
# time_series_data_lat = time_series_data_lat.iloc[:, 0:lim_time_idx]
# time_series_data_vel = time_series_data_vel.iloc[:, 0:lim_time_idx]
# time_series_data_hdg = time_series_data_hdg.iloc[:, 0:lim_time_idx]
# time_series_data_vrate = time_series_data_vrate.iloc[:, 0:lim_time_idx]
# time_series_data_accel = time_series_data_accel.iloc[:, 0:lim_time_idx]
# time_series_data_turnrate = time_series_data_turnrate.iloc[:, 0:lim_time_idx]
cols_list = list(time_series_data.columns)

# # drop all cols that contain only nan values for all rows ---------------------------- used in old version
# for col in cols_list:
#     temp_df = time_series_data[col]
#     # print('------------------')
#     # print('col = ', col)
#     # print('temp_df = \n', temp_df)
#     # print('nans = ', temp_df.isnull().values.all())
#     if temp_df.isnull().values.all() == True:
#         print('--> dropping', str(col))
#         time_series_data = time_series_data.drop(str(col), axis=1) #------------------------ used in old version (up to top)

# print('time_series_data 1 =\n', time_series_data)

print('processed time_series_data =\n', time_series_data)

# plt.plot(undersample_data.datetime,undersample_data['Vancouver'])
# plt.xlabel('Time (year)',fontsize=20)
# plt.ylabel('Temperature',fontsize=20)
# # plt.show()


# transform time series data to array format
# remove datetime col
# data_array = np.array(undersample_data.T.drop('datetime').values)
# data_array = np.array(new_time_series_data.drop('datetime', axis=1).values)
# data_array = np.array(time_series_data.T.drop('datetime').values) #---------------------------------------- old version
# data_array = np.array(time_series_data.T.values)
# data_array = np.array(time_series_data.values)
# print('data_array 1 = \n', data_array)
# print('data_array 1 shape = \n', data_array.shape)

# multidim_data_array =
#
# # print('multidimensional data_array = \n', multidim_data_array)
# # print('multidimensional data_array shape = \n', multidim_data_array.shape)
# test_array1 = np.array(time_series_data.T.values)
# print('\ntest array 1 = \n', test_array1)
# print('test array shape 1 = \n', test_array1.shape)
#
# test_array2 = np.array(time_series_data.values)
# print('\ntest array 2 = \n', test_array2)
# print('test array shape 2 = \n', test_array2.shape)
#
# # lon, lat, alt
# test_array3 = np.stack([np.array(time_series_data_lon.T.values),
#                         np.array(time_series_data_lat.T.values),
#                         np.array(time_series_data.T.values)], axis=1)
# print('\ntest array 3 = \n', test_array3)
# print('test array shape 3 = \n', test_array3.shape)
#
# # lon, lat, alt, vel, hdg
# test_array4 = np.stack([np.array(time_series_data_lon.T.values),
#                         np.array(time_series_data_lat.T.values),
#                         np.array(time_series_data.T.values),
#                         np.array(time_series_data_vel.T.values),
#                         np.array(time_series_data_hdg.T.values),
#                         ], axis=1)
# print('\ntest array 4 = \n', test_array4)
# print('test array shape 4 = \n', test_array4.shape)
#
# # lon, lat, alt, vel, hdg, vrate
# test_array5 = np.stack([np.array(time_series_data_lon.T.values),
#                         np.array(time_series_data_lat.T.values),
#                         np.array(time_series_data.T.values),
#                         np.array(time_series_data_vel.T.values),
#                         np.array(time_series_data_hdg.T.values),
#                         np.array(time_series_data_vrate.T.values),
#                         ], axis=1)
# print('\ntest array 5 = \n', test_array5)
# print('test array shape 5 = \n', test_array5.shape)
#
# # vel, hdg, vrate
# test_array6 = np.stack([np.array(time_series_data_vel.T.values),
#                         np.array(time_series_data_hdg.T.values),
#                         np.array(time_series_data_vrate.T.values),
#                         ], axis=1)
# print('\ntest array 6 = \n', test_array6)
# print('test array shape 6 = \n', test_array6.shape)
#
# # lon, lat, alt, vrate
# test_array7 = np.stack([np.array(time_series_data_lon.T.values),
#                         np.array(time_series_data_lat.T.values),
#                         np.array(time_series_data.T.values),
#                         np.array(time_series_data_vrate.T.values),
#                         ], axis=1)
# print('\ntest array 7 = \n', test_array5)
# print('test array shape 7 = \n', test_array5.shape)


# features_str_list = ['alt']
features_str_list = ['lon', 'lat', 'alt']
# features_str_list = ['lon', 'lat', 'alt', 'hdg']
# features_str_list = ['lon', 'lat', 'alt', 'vel', 'hdg', 'vrate']
# features_str_list = ['lon', 'lat', 'alt', 'vel', 'turnrate', 'acceleration', 'vrate']
# features_str_list = ['lon', 'lat', 'alt', 'turnrate', 'acceleration', 'vrate']
# features_str_list = ['turnrate', 'acceleration', 'vrate']
# features_str_list = ['vel', 'turnrate', 'acceleration', 'vrate']
import collections
print('features = ', features_str_list)
if collections.Counter(features_str_list) == collections.Counter(['alt']):
    data_array = np.array(time_series_data.values)
if collections.Counter(features_str_list) == collections.Counter(['lon', 'lat', 'alt']):
    data_array = np.stack([np.array(time_series_data_lon.T.values),
                        np.array(time_series_data_lat.T.values),
                        np.array(time_series_data.T.values)], axis=1)
if collections.Counter(features_str_list) == collections.Counter(['lon', 'lat', 'alt', 'hdg']):
    data_array = np.stack([np.array(time_series_data_lon.T.values),
                        np.array(time_series_data_lat.T.values),
                        np.array(time_series_data.T.values),
                        np.array(time_series_data_hdg.T.values)], axis=1)
if collections.Counter(features_str_list) == collections.Counter(['lon', 'lat', 'alt', 'vel', 'hdg']):
    data_array = np.stack([np.array(time_series_data_lon.T.values),
                        np.array(time_series_data_lat.T.values),
                        np.array(time_series_data.T.values),
                        np.array(time_series_data_vel.T.values),
                        np.array(time_series_data_hdg.T.values),
                        ], axis=1)
if collections.Counter(features_str_list) == collections.Counter(['lon', 'lat', 'alt', 'vel', 'hdg', 'vrate']):
    data_array = np.stack([np.array(time_series_data_lon.T.values),
                        np.array(time_series_data_lat.T.values),
                        np.array(time_series_data.T.values),
                        np.array(time_series_data_vel.T.values),
                        np.array(time_series_data_hdg.T.values),
                        np.array(time_series_data_vrate.T.values),
                        ], axis=1)
if collections.Counter(features_str_list) == collections.Counter(['vel', 'hdg', 'vrate']):
    data_array = np.stack([np.array(time_series_data_vel.T.values),
                        np.array(time_series_data_hdg.T.values),
                        np.array(time_series_data_vrate.T.values),
                        ], axis=1)
if collections.Counter(features_str_list) == collections.Counter(['lon', 'lat', 'alt', 'vrate']):
    data_array = np.stack([np.array(time_series_data_lon.T.values),
                        np.array(time_series_data_lat.T.values),
                        np.array(time_series_data.T.values),
                        np.array(time_series_data_vrate.T.values),
                        ], axis=1)
if collections.Counter(features_str_list) == collections.Counter(['lon', 'lat', 'alt', 'vel', 'turnrate', 'acceleration', 'vrate']):
    data_array = np.stack([np.array(time_series_data_lon.T.values),
                        np.array(time_series_data_lat.T.values),
                        np.array(time_series_data.T.values),
                        np.array(time_series_data_vel.T.values),
                        np.array(time_series_data_accel.T.values),
                        np.array(time_series_data_turnrate.T.values),
                        np.array(time_series_data_vrate.T.values),
                        ], axis=1)
if collections.Counter(features_str_list) == collections.Counter(['lon', 'lat', 'alt', 'turnrate', 'acceleration', 'vrate']):
    data_array = np.stack([np.array(time_series_data_lon.T.values),
                        np.array(time_series_data_lat.T.values),
                        np.array(time_series_data.T.values),
                        np.array(time_series_data_accel.T.values),
                        np.array(time_series_data_turnrate.T.values),
                        np.array(time_series_data_vrate.T.values),
                        ], axis=1)
if collections.Counter(features_str_list) == collections.Counter(['turnrate', 'acceleration', 'vrate']):
    data_array = np.stack([
                        np.array(time_series_data_accel.T.values),
                        np.array(time_series_data_turnrate.T.values),
                        np.array(time_series_data_vrate.T.values),
                        ], axis=1)
if collections.Counter(features_str_list) == collections.Counter(['vel','turnrate', 'acceleration', 'vrate']):
    data_array = np.stack([
                        np.array(time_series_data_vel.T.values),
                        np.array(time_series_data_accel.T.values),
                        np.array(time_series_data_turnrate.T.values),
                        np.array(time_series_data_vrate.T.values),
                        ], axis=1)
# standardize the data_array to values between -1 and 1
# print(data_array.shape)
# print(data_array.min().round(5), data_array.max().round(5)) # -20, 100
scaler = MinMaxScaler(feature_range=(-1,1))
# scaler = MinMaxScaler(feature_range=(0,1))
data_array = scaler.fit_transform(data_array.reshape(data_array.shape[0], -1)).reshape(data_array.shape)
# print(data_array.shape)
# print(data_array.min().round(5), data_array.max().round(5)) # -1, 1

# print('data_array shape = \n', data_array.shape)
# data_array = resample(data_array, 500)  # for debugging
# print('data_array = \n', data_array)
# print('data_array shape = \n', data_array.shape)

# # data_array = data_array[:, ~np.isnan(data_array).any(axis=0)]
# # data_array = data_array[:, ~pd.isnull(data_array).any(axis=0)]
# # data_array = data_array[:, ~pd.isnull(data_array)]
# # data_array = data_array[np.logical_not(np.isnan(data_array))]
# # data_array = data_array[np.logical_not(pd.isnull(data_array))]
# print(pd.isnull(data_array))
# print(np.any(pd.isnull(data_array)))


# new_data_array = np.array([])
# array_idx = 0
# for row in data_array:
#     temp_row_array = np.array([])
#     print('-----------------------------------')
#     print('row = ', row)
#     for element in row:
#         # print('element = ', element)
#         # print('element type = ', type(element))
#         # if element != np.nan:
#         if np.isnan(element) != True:
#             # print('!!! not nan detected !!!')
#             temp_row_array = np.append(temp_row_array, element)
#             # np.append(element, temp_row_array)
#         # else:
#         #     print('--> nan detected')
#         # print('temp_row_array =\n', temp_row_array)
#     # new_data_array = np.expand_dims(new_data_array, axis=1)
#     # new_data_array = np.append(new_data_array, temp_row_array)
#     # new_data_array = np.append(new_data_array, temp_row_array, axis=0)
#     if np.size(temp_row_array) != 0:
#         print('temp_row_array not empty')
#         print('temp_row_array =\n', temp_row_array)
#         # new_data_array = np.vstack((new_data_array, temp_row_array))
#         new_data_array = np.array([new_data_array, temp_row_array])
#     print('new_data_array =\n', new_data_array)
#     array_idx += 1


# # print('new_data_array = \n', new_data_array)
# print('new_data_array = \n', new_data_array.shape)
# print('data_array 2 = \n', data_array)

# X = np.array([-0.070024, -0.011244, -0.048864])
# Y = np.array([-0.046507, -0.032194, 0.065276])
# Z = np.array([0.065012, 0.078344])
# from tslearn.utils import to_time_series_dataset

# data_array = to_time_series_dataset([X,Y,Z])
# print('data_array = \n', data_array)

# icao_list = time_series_data.T.drop('datetime').index.tolist()
# track_id_list = time_series_data.T.drop('datetime').index.tolist() ----------------------------- old version
track_id_list = time_series_data.T.index.tolist()
# print('icao_list =\n', icao_list)
ncluster_range_mid = int(math.sqrt(len(track_id_list)))
ncluster_range_min = int(ncluster_range_mid-(ncluster_range_mid*0.1))
ncluster_range_max = int(ncluster_range_mid+(ncluster_range_mid*0.1))
print('len track_id_list = ', len(track_id_list))
# print('ncluster_range_min = ', ncluster_range_min)
# print('ncluster_range_mid = ', ncluster_range_mid)
# print('ncluster_range_max = ', ncluster_range_max)


# import tslearn's time series version of KMeans algorithm
from tslearn.clustering import TimeSeriesKMeans
# from tslearn.clustering import KShape
# from sklearn.cluster import DBSCAN

# build model
# n_clusters = int(math.sqrt(len(track_id_list)))
# n_clusters = 5
# n_clusters_range = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50]
# n_clusters_range = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]
# n_clusters_range = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
# n_clusters_range = [2, 3, 4]  # for debugging
n_clusters_range = [30]  # for debugging
# print('n_clusters = ', n_clusters)

# # clustering algorithm #1: KMeans with DTW
# model = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", max_iter=10)
# # model = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", max_iter=10, n_jobs=-1, verbose=1)
# # model = TimeSeriesKMeans(n_clusters=n_clusters, max_iter=10, n_jobs=-1, verbose=1)
# # model = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", max_iter=2, max_iter_barycenter=10, verbose=1)
# # model = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", max_iter=2, max_iter_barycenter=10, verbose=1, n_jobs=-1)
# # model = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", max_iter=2, max_iter_barycenter=2, verbose=1, n_jobs=1)

# clustering algorithm #2: Shapelet (also called KShape)
# model = KShape(n_clusters=n_clusters, max_iter=10)

# # clustering algorithm #3: DBSCAN
# epsilon = .5
# min_samples = 3
# model = DBSCAN(eps=epsilon, min_samples=min_samples)

# # # fit the data to the built model
# # nsamples, nx, ny = data_array.shape # for DBSCAN only
# # d2_train_dataset = data_array.reshape((nsamples,nx*ny)) # for DBSCAN only
# # data_array = d2_train_dataset # for DBSCAN only
# model.fit(data_array)
# # labels = model.labels_ # for DBSCAN only
# # no_clusters = len(np.unique(labels)) # for DBSCAN only
# # no_noise = np.sum(np.array(labels) == -1, axis=0) # for DBSCAN only
# # print('Number of detected clusters = ', no_clusters)
# # print('Number of outliers          = ', no_noise, '/', len(labels), '(', round(100*no_noise/len(labels), 2),'%)')
#
# executionTime1 = (time.time() - startTime)
# print('Execution time in seconds: ' + str(round(executionTime1, 2)))
#
# labels = model.labels_
# cluster_centers = model.cluster_centers_
# inertia = model.inertia_
# n_iter = model.n_iter_
# print('labels = \n', labels)
# print('labels shape = \n', labels.shape)
# print('cluster_centers = \n', cluster_centers)
# print('cluster_centers shape = \n', cluster_centers.shape)
# print('inertia = ', inertia)
# print('n_iter = ', n_iter)
#
# # apply the fitted model to our dataset
# y = model.predict(data_array)
#
# # calculate the silhouette score
# # # version #1 using sklearn (needs 2D flattening and doesnt work for dtw...)
# # from sklearn.metrics import silhouette_score
# # nsamples, nx, ny = data_array.shape # used to flatten > 2D array to 2D
# # d2_train_dataset = data_array.reshape((nsamples,nx*ny)) # used to flatten > 2D array to 2D
# # data_array = d2_train_dataset # used to flatten > 2D array to 2D
# # s_score = silhouette_score(data_array, y, metric='euclidean')
# # version #2 using tslearn
# from tslearn.clustering import silhouette_score
# # s_score = silhouette_score(data_array, y, metric='dtw', sample_size=100)
# # print('silhouette score = ', s_score)
#
# executionTime2 = (time.time() - startTime)
# print('Execution time in seconds: ' + str(round(executionTime2, 2)))

silhouette_avg, sum_of_squared_distances = [], []
def multi_cluster_model(n_clusters, data=data_array):
    startTime = time.time()

    # build model
    # n_clusters = int(math.sqrt(len(track_id_list)))
    # n_clusters = 30
    # print('--------------------------------------')
    # print('n_clusters = ', n_clusters)
    # print('... creating the model')

    # clustering algorithm #1: KMeans with DTW
    model = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw")
    # model = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", max_iter=10)
    # model = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", max_iter=10, n_jobs=-1, verbose=1)
    # model = TimeSeriesKMeans(n_clusters=n_clusters, max_iter=10, n_jobs=-1, verbose=1)
    # model = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", max_iter=2, max_iter_barycenter=10, verbose=1)
    # model = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", max_iter=2, max_iter_barycenter=10, verbose=1, n_jobs=-1)
    # model = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", max_iter=2, max_iter_barycenter=2, verbose=1, n_jobs=1)

    # clustering algorithm #2: Shapelet (also called KShape)
    # model = KShape(n_clusters=n_clusters, max_iter=10)

    # # clustering algorithm #3: DBSCAN
    # epsilon = .5
    # min_samples = 3
    # model = DBSCAN(eps=epsilon, min_samples=min_samples)
    # executionTime2 = (time.time() - startTime)
    # print('Execution time in seconds: ' + str(round(executionTime2, 2)))

    # # fit the data to the built model
    # nsamples, nx, ny = data_array.shape # for DBSCAN only
    # d2_train_dataset = data_array.reshape((nsamples,nx*ny)) # for DBSCAN only
    # data_array = d2_train_dataset # for DBSCAN only
    # print('... fitting the model')
    model.fit(data)
    # labels = model.labels_ # for DBSCAN only
    # no_clusters = len(np.unique(labels)) # for DBSCAN only
    # no_noise = np.sum(np.array(labels) == -1, axis=0) # for DBSCAN only
    # print('Number of detected clusters = ', no_clusters)
    # print('Number of outliers          = ', no_noise, '/', len(labels), '(', round(100*no_noise/len(labels), 2),'%)')

    model_fit_time = (time.time() - startTime)
    print('model_fit_time [s]       = ' + str(round(model_fit_time, 2)))

    cluster_labels = model.labels_
    cluster_intertia = model.inertia_
    cluster_centers = model.cluster_centers_
    # print('cluster_centers = ', cluster_centers)
    # print('cluster_centers shape = ', cluster_centers.shape)

    interpret_results_WCSS_Minimizers(features=features_str_list, centroids=cluster_centers, path=results_path)

    ssd = cluster_intertia
    sum_of_squared_distances.append(ssd)
    print('sum of squared distances = ', round(ssd, 4))

    model_preds = model.predict(data)

    # labels = model.labels_
    # cluster_centers = model.cluster_centers_
    # inertia = model.inertia_
    # n_iter = model.n_iter_
    # print('labels = \n', labels)
    # print('labels shape = \n', labels.shape)
    # print('cluster_centers = \n', cluster_centers)
    # print('cluster_centers shape = \n', cluster_centers.shape)
    # print('inertia = ', inertia)
    # print('n_iter = ', n_iter)

    # # apply the fitted model to our dataset
    # print('... performing model predictions')
    # y = model.predict(data_array)

    # prediction_time = (time.time() - startTime)
    # print('Execution time in seconds: ' + str(round(prediction_time, 2)))

    # calculate the silhouette score
    print('... calculating silhouette score')
    # # version #1 using sklearn (needs 2D flattening and doesnt work for dtw...)
    # from sklearn.metrics import silhouette_score
    # nsamples, nx, ny = data_array.shape # used to flatten > 2D array to 2D
    # d2_train_dataset = data_array.reshape((nsamples,nx*ny)) # used to flatten > 2D array to 2D
    # data_array = d2_train_dataset # used to flatten > 2D array to 2D
    # s_score = silhouette_score(data_array, y, metric='euclidean')
    # version #2 using tslearn
    # s_score = silhouette_score(data_array, y, metric='dtw', sample_size=100)
    # s_score = silhouette_score(data_array, y, metric='softdtw', sample_size=100)
    # s_score = silhouette_score(data_array, cluster_labels, metric='dtw')
    s_score = 1
    print('silhouette score         = ', round(s_score, 4))
    silhouette_avg.append(s_score)

    s_score_calc_time = (time.time() - startTime)
    print('s_score_calc_time [s]    = ' + str(round(s_score_calc_time, 2)))

    nsamples, nx, ny = data.shape # used to flatten > 2D array to 2D
    d2_train_dataset = data.reshape((nsamples, nx*ny)) # used to flatten > 2D array to 2D
    flat_data_array = d2_train_dataset # used to flatten > 2D array to 2D
    silhouette_vals = silhouette_samples(flat_data_array, model_preds)

    return n_clusters, s_score, ssd, model_fit_time, s_score_calc_time, cluster_labels, cluster_centers, model_preds, silhouette_vals
    # return n_clusters, s_score, ssd, model_fit_time, s_score_calc_time, cluster_labels, cluster_centers, model_preds

# same as multivariate_timeseries_plot below but only works for 1 feature (cannot generate subplots)
def multivariate_timeseries_plot_nosubplots(multitimeseries_data, model_preds, centroids, cluster_idx, path, ax=None, plt_kwargs={}):
    # temp_data = data[data['icao24'] == ac_id]

    cluster_idx = cluster_idx-1
    print('cluster_idx = ', cluster_idx)
    print('model_preds = ', model_preds)


    # print('centroids   = ', centroids)
    # print('centroids.shape   = ', centroids.shape)
    # print('centroids[cluster_idx]   = ', centroids)
    print('centroids[cluster_idx].shape   = ', centroids[cluster_idx].shape)
    alt_centroids = centroids[cluster_idx][0]
    lon_centroids = centroids[cluster_idx][1]
    lat_centroids = centroids[cluster_idx][2]
    # print('alt_centroids = ', alt_centroids)
    # print('lon_centroids = ', lon_centroids)
    # print('lat_centroids = ', lat_centroids)
    # print('alt_centroids.shape = ', alt_centroids.shape)
    # print('lon_centroids.shape = ', lon_centroids.shape)
    # print('lat_centroids.shape = ', lat_centroids.shape)

    cluster_icao_idx = np.where(model_preds==cluster_idx)[0]
    # print('cluster_icao_idx = ', cluster_icao_idx)
    # track_id_array = np.array(track_id_list)
    # print('track_id in cluster = ', track_id_array[np.where(y==cluster_idx)[0]])
    # print('cluster centroid = \n', cluster_centers[cluster_idx])
    print('nb of traj in cluster = ', len(cluster_icao_idx))

    if ax is None:
        # fig = plt.gcf().set_size_inches(10, 5)
        fig = plt.gcf()
        ax = plt.gca()
    # for icao in icao_array[np.where(y==cluster_idx)[0]]:
    for icao_idx in cluster_icao_idx:
        # print('======================================')
        # print('icao_idx = ', icao_idx)
        # print('len(cluster_icao_idx) = ', len(cluster_icao_idx))
        # print('cluster_icao_idx[-1] = ', cluster_icao_idx[-1])

        # print('multitimeseries_data[icao_idx] = ', multitimeseries_data[icao_idx])
        # print('multitimeseries_data[icao_idx].shape = ', multitimeseries_data[icao_idx].shape)
        # print('multitimeseries_data[icao_idx][0] = ', multitimeseries_data[icao_idx][0])
        # print('multitimeseries_data[icao_idx][1] = ', multitimeseries_data[icao_idx][1])
        # print('multitimeseries_data[icao_idx][2] = ', multitimeseries_data[icao_idx][2])

        alt_data_for_icao_idx = multitimeseries_data[icao_idx][0]
        lon_data_for_icao_idx = multitimeseries_data[icao_idx][1]
        lat_data_for_icao_idx = multitimeseries_data[icao_idx][2]
        # print('alt_data_for_icao_idx = ', alt_data_for_icao_idx)
        # print('alt_data_for_icao_idx.shape = ', alt_data_for_icao_idx.shape)
        # print('lon_data_for_icao_idx = ', lon_data_for_icao_idx)
        # print('lon_data_for_icao_idx.shape = ', lon_data_for_icao_idx.shape)
        # print('lat_data_for_icao_idx = ', lat_data_for_icao_idx)
        # print('lat_data_for_icao_idx.shape = ', lat_data_for_icao_idx.shape)

        # ax.plot(x_data_array, data[icao_idx], **plt_kwargs, label=str(track_id_array[icao_idx]))
        # ax.scatter(x_data_array, data[icao_idx], **plt_kwargs, label=str(icao_array[icao_idx]))
        # ax.plot(range(0, len(data[icao_idx])), data[icao_idx], alpha=0.2, **plt_kwargs, label=str(track_id_array[icao_idx]))

        # used to print on plot the 'trajectory' legend only for the last trajectory instead of for all
        if icao_idx != cluster_icao_idx[-1]:
            # print('trigger 1')
            ax.plot(range(0, len(alt_data_for_icao_idx)), alt_data_for_icao_idx, c='gray', alpha=0.3, **plt_kwargs)
        else:
            # print('trigger 2')
            ax.plot(range(0, len(alt_data_for_icao_idx)), alt_data_for_icao_idx, c='gray', alpha=0.3, label='trajectories', **plt_kwargs)
    ax.plot(range(0, len(alt_data_for_icao_idx)), alt_centroids, c='red', label='centroid', **plt_kwargs)
    ax.legend(loc='upper right')
    ax.annotate(str(len(cluster_icao_idx)), xy=(1, 0), xycoords='axes fraction', fontsize=15, horizontalalignment='right', verticalalignment='bottom')
    fig.suptitle('Cluster '+str(cluster_idx))
    ax.set_xlabel('Time')
    ax.set_ylabel('Altitude [m]')
    fig_name = path + "alt_cluster_" + str(cluster_idx) + "_of_" + str(n_clusters) + ".png"
    fig.savefig(fig_name, dpi=200)
    fig.clear(ax)
    plt.close(fig)
    # return ax


def multivariate_timeseries_plot(multitimeseries_data, model_preds, centroids, cluster_idx, path, ax=None, plt_kwargs={}):
    # temp_data = data[data['icao24'] == ac_id]

    # print('model_preds = ', model_preds)
    # print('model_preds.shape = ', model_preds.shape)

    for i, cluster in enumerate(np.unique(model_preds)):
        # print('=================================================')
        print('     --> plotting cluster', str(i),'(from',str(cluster_idx),'total clusters)')
        # print('=================================================')

        # print('cluster name = ', i)
        # cluster_idx = cluster_idx-1
        # cluster_idx = cluster
        # print('cluster     = ', cluster_id)
        # print('cluster_idx = ', cluster_idx)
        # print('model_preds = ', model_preds)

        # print('centroids   = ', centroids)
        # print('centroids.shape   = ', centroids.shape)
        # print('centroids[cluster_idx]   = ', centroids)
        # print('centroids[cluster_idx].shape   = ', centroids[cluster_idx].shape)
        # alt_centroids = centroids[cluster_idx][0]
        # lon_centroids = centroids[cluster_idx][1]
        # lat_centroids = centroids[cluster_idx][2]
        alt_centroids = centroids[i][0]
        lon_centroids = centroids[i][1]
        lat_centroids = centroids[i][2]
        # print('alt_centroids = ', alt_centroids)
        # print('lon_centroids = ', lon_centroids)
        # print('lat_centroids = ', lat_centroids)
        # print('alt_centroids.shape = ', alt_centroids.shape)
        # print('lon_centroids.shape = ', lon_centroids.shape)
        # print('lat_centroids.shape = ', lat_centroids.shape)

        # get individual trajectories that are in the cluster i and return their indexes
        # cluster_icao_idx = np.where(model_preds == cluster_idx)[0]
        cluster_icao_idx = np.where(model_preds == i)[0]
        # print('cluster_icao_idx = ', cluster_icao_idx)
        # track_id_array = np.array(track_id_list)
        # print('track_id in cluster = ', track_id_array[np.where(y==cluster_idx)[0]])
        # print('cluster centroid = \n', cluster_centers[cluster_idx])
        # print('nb of traj in cluster', str(i),'= ', len(cluster_icao_idx))

        if ax is None:
            # fig = plt.gcf().set_size_inches(10, 5)
            fig = plt.gcf()
            # ax = plt.gca()
            # fig, axs = plt.subplots(1, 3)
            # axs = plt.subplots(1, 3)
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

        # for icao in icao_array[np.where(y==cluster_idx)[0]]:
        for icao_idx in cluster_icao_idx:
            # print('+++    _idx = ', icao_idx)
            # print('len(cluster_icao_idx) = ', len(cluster_icao_idx))
            # print('cluster_icao_idx[-1] = ', cluster_icao_idx[-1])

            # print('multitimeseries_data[icao_idx] = ', multitimeseries_data[icao_idx])
            # print('multitimeseries_data[icao_idx].shape = ', multitimeseries_data[icao_idx].shape)
            # print('multitimeseries_data[icao_idx][0] = ', multitimeseries_data[icao_idx][0])
            # print('multitimeseries_data[icao_idx][1] = ', multitimeseries_data[icao_idx][1])
            # print('multitimeseries_data[icao_idx][2] = ', multitimeseries_data[icao_idx][2])

            alt_data_for_icao_idx = multitimeseries_data[icao_idx][0]
            lon_data_for_icao_idx = multitimeseries_data[icao_idx][1]
            lat_data_for_icao_idx = multitimeseries_data[icao_idx][2]
            # print('alt_data_for_icao_idx = ', alt_data_for_icao_idx)
            # print('alt_data_for_icao_idx.shape = ', alt_data_for_icao_idx.shape)
            # print('lon_data_for_icao_idx = ', lon_data_for_icao_idx)
            # print('lon_data_for_icao_idx.shape = ', lon_data_for_icao_idx.shape)
            # print('lat_data_for_icao_idx = ', lat_data_for_icao_idx)
            # print('lat_data_for_icao_idx.shape = ', lat_data_for_icao_idx.shape)

            # ax.plot(x_data_array, data[icao_idx], **plt_kwargs, label=str(track_id_array[icao_idx]))
            # ax.scatter(x_data_array, data[icao_idx], **plt_kwargs, label=str(icao_array[icao_idx]))
            # ax.plot(range(0, len(data[icao_idx])), data[icao_idx], alpha=0.2, **plt_kwargs, label=str(track_id_array[icao_idx]))

            # used to print on plot the 'trajectory' legend only for the last trajectory instead of for all
            opacity = 0.6
            if icao_idx != cluster_icao_idx[-1]:
                # print('trigger 1')
                # ax.plot(range(0, len(alt_data_for_icao_idx)), alt_data_for_icao_idx, c='gray', alpha=0.3, **plt_kwargs)
                # axs[0, 0].plot(range(0, len(alt_data_for_icao_idx)), alt_data_for_icao_idx, c='gray', alpha=0.3, **plt_kwargs)
                # axs[0, 1].plot(range(0, len(alt_data_for_icao_idx)), lon_data_for_icao_idx, c='gray', alpha=0.3, **plt_kwargs)
                # axs[0, 2].plot(range(0, len(alt_data_for_icao_idx)), lat_data_for_icao_idx, c='gray', alpha=0.3, **plt_kwargs)
                ax1.plot(range(0, len(alt_data_for_icao_idx)), alt_data_for_icao_idx, c='gray', alpha=opacity, **plt_kwargs)
                ax2.plot(range(0, len(alt_data_for_icao_idx)), lon_data_for_icao_idx, c='gray', alpha=opacity, **plt_kwargs)
                ax3.plot(range(0, len(alt_data_for_icao_idx)), lat_data_for_icao_idx, c='gray', alpha=opacity, **plt_kwargs)
            else:
                # print('trigger 2')
                # ax.plot(range(0, len(alt_data_for_icao_idx)), alt_data_for_icao_idx, c='gray', alpha=0.3, label='trajectories', **plt_kwargs)
                # axs[0, 0].plot(range(0, len(alt_data_for_icao_idx)), alt_data_for_icao_idx, c='gray', alpha=0.3, label='trajectories', **plt_kwargs)
                # axs[0, 1].plot(range(0, len(alt_data_for_icao_idx)), lon_data_for_icao_idx, c='gray', alpha=0.3, label='trajectories', **plt_kwargs)
                # axs[0, 2].plot(range(0, len(alt_data_for_icao_idx)), lat_data_for_icao_idx, c='gray', alpha=0.3, label='trajectories', **plt_kwargs)
                ax1.plot(range(0, len(alt_data_for_icao_idx)), alt_data_for_icao_idx, c='gray', alpha=opacity, label='trajectories', **plt_kwargs)
                ax2.plot(range(0, len(alt_data_for_icao_idx)), lon_data_for_icao_idx, c='gray', alpha=opacity, label='trajectories', **plt_kwargs)
                ax3.plot(range(0, len(alt_data_for_icao_idx)), lat_data_for_icao_idx, c='gray', alpha=opacity, label='trajectories', **plt_kwargs)
        # ax.plot(range(0, len(alt_data_for_icao_idx)), alt_centroids, c='red', label='centroid', **plt_kwargs)
        ax1.plot(range(0, len(alt_data_for_icao_idx)), alt_centroids, c='red', linewidth=2,label='centroid', **plt_kwargs)
        ax2.plot(range(0, len(lon_data_for_icao_idx)), lon_centroids, c='red', linewidth=2, label='centroid', **plt_kwargs)
        ax3.plot(range(0, len(lat_data_for_icao_idx)), lat_centroids, c='red', linewidth=2, label='centroid', **plt_kwargs)

        # fig.suptitle('Cluster '+str(cluster_idx)+' out of '+str(cluster))
        fig.suptitle('Cluster ID '+str(i)+' out of '+str(cluster_idx))
        ax3.legend(loc='upper right')
        # ax3.annotate(str(len(cluster_icao_idx)), xy=(1, 0), xycoords='axes fraction', fontsize=15, horizontalalignment='right', verticalalignment='bottom')

        ax1.title.set_text('Altitude')
        ax2.title.set_text('Longitude')
        ax3.title.set_text('Latitude')

        # ax.set_xlabel('Time')
        # ax.set_ylabel('Altitude [m]')
        ax2.set_xlabel('Time Index')
        ax1.set_ylabel('Normalized Values')
        # ax2.axes.get_yaxis().set_visible(False)
        # ax3.axes.get_yaxis().set_visible(False)
        # fig.subplots_adjust(wspace=0, hspace=0)

        fig_name = path + "nb clusters "+str(cluster_idx)+" features vs time cluster " + str(i) +".png"
        fig.savefig(fig_name, dpi=200)
        fig.clear(ax)
        # fig.clear(axs)
        fig.clear(ax1)
        fig.clear(ax2)
        fig.clear(ax3)
        plt.close(fig)
        ax = None

import random
from matplotlib.pyplot import cm
get_colors = lambda n: ["#%06x" % random.randint(0, 0xFFFFFF) for _ in range(n)]
# get_colors = iter(cm.rainbow(np.linspace(0, 1, n)))
# get_colors = lambda n: [cm.rainbow(n) for _ in range(n)]

def multivariate_timeseries_plot2(multitimeseries_data, model_preds, ac_info_data,centroids, color_list, cluster_idx, path, features, ax=None, plt_kwargs={}):
    # temp_data = data[data['icao24'] == ac_id]

    # print('model_preds = ', model_preds)
    # print('model_preds.shape = ', model_preds.shape)

    # print('centroids = ', centroids)
    # print('centroids.shape = ', centroids.shape)
    # print('len(centroids) = ', len(centroids))

    # print('features = ', features)
    # print('len(features) = ', len(features))

    # # make a list of random colors for each unique cluster
    # color_list = get_colors(cluster_idx)
    # color_list = cm.rainbow(np.linspace(0, 1, cluster_idx))
    # color_list = cm.Set2(np.linspace(0, 1, cluster_idx))
    # print('color_list = ', color_list)

    ac_info_data = ac_info_data.astype({'time': 'float', 'lat': 'float', 'lon': 'float', 'geoaltitude': 'float',
                                        'heading': 'float', 'turnrate': 'float', 'acceleration': 'float'})

    if features == ['lon', 'lat', 'alt']:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

    if features == ['lon', 'lat', 'alt', 'hdg']:
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)

    for i, cluster in enumerate(np.unique(model_preds)):
        # print('=================================================')
        print('     --> plotting cluster', str(i),'(from',str(cluster_idx),'total clusters)')
        # print('=================================================')

        # print('cluster name = ', i)
        # cluster_idx = cluster_idx-1
        # cluster_idx = cluster
        # print('cluster     = ', cluster_id)
        # print('cluster     = ', cluster)
        # print('cluster_idx = ', cluster_idx)
        # print('model_preds = ', model_preds)

        # print('centroids   = ', centroids)
        # print('centroids.shape   = ', centroids.shape)
        # print('centroids[cluster_idx]   = ', centroids)
        # print('centroids[cluster_idx].shape   = ', centroids[cluster_idx].shape)
        # alt_centroids = centroids[cluster_idx][0]
        # lon_centroids = centroids[cluster_idx][1]
        # lat_centroids = centroids[cluster_idx][2]

        ac_info_data_for_cluster = ac_info_data[ac_info_data["cluster"] == cluster]

        if features == ['lon', 'lat', 'alt']:
            lon_centroids = centroids[i][0]
            lat_centroids = centroids[i][1]
            alt_centroids = centroids[i][2]
            # lon_centroids = ac_info_data_for_cluster.groupby(['time'])['lon'].mean().tolist()
            # lat_centroids = ac_info_data_for_cluster.groupby(['time'])['lat'].mean().tolist()
            # alt_centroids = ac_info_data_for_cluster.groupby(['time'])['geoaltitude'].mean().tolist()

        if features == ['lon', 'lat', 'alt', 'hdg']:
            lon_centroids = centroids[i][0]
            lat_centroids = centroids[i][1]
            alt_centroids = centroids[i][2]
            hdg_centroids = centroids[i][3]
        # print('alt_centroids = ', alt_centroids)
        # print('lon_centroids = ', lon_centroids)
        # print('lat_centroids = ', lat_centroids)
        # print('alt_centroids.shape = ', alt_centroids.shape)
        # print('lon_centroids.shape = ', lon_centroids.shape)
        # print('lat_centroids.shape = ', lat_centroids.shape)
        # print('     nb in cluster = ', len(lon_centroids))

        # get individual trajectories that are in the cluster i and return their indexes
        # cluster_icao_idx = np.where(model_preds == cluster_idx)[0]
        cluster_icao_idx = np.where(model_preds == i)[0]
        # print('cluster_icao_idx = ', cluster_icao_idx)
        # track_id_array = np.array(track_id_list)
        # print('track_id in cluster = ', track_id_array[np.where(y==cluster_idx)[0]])
        # print('cluster centroid = \n', cluster_centers[cluster_idx])
        # print('nb of traj in cluster', str(i),'= ', len(cluster_icao_idx))


        # if ax is None:
        #     # fig = plt.gcf().set_size_inches(10, 5)
        #     fig = plt.gcf()
        #     # ax = plt.gca()
        #     # fig, axs = plt.subplots(1, 3)
        #     # axs = plt.subplots(1, 3)
        #     fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

        # for icao in icao_array[np.where(y==cluster_idx)[0]]:
        for icao_idx in cluster_icao_idx:
            # print('+++    _idx = ', icao_idx)
            # print('len(cluster_icao_idx) = ', len(cluster_icao_idx))
            # print('cluster_icao_idx[-1] = ', cluster_icao_idx[-1])

            # print('multitimeseries_data[icao_idx] = ', multitimeseries_data[icao_idx])
            # print('multitimeseries_data[icao_idx].shape = ', multitimeseries_data[icao_idx].shape)
            # print('multitimeseries_data[icao_idx][0] = ', multitimeseries_data[icao_idx][0])
            # print('multitimeseries_data[icao_idx][1] = ', multitimeseries_data[icao_idx][1])
            # print('multitimeseries_data[icao_idx][2] = ', multitimeseries_data[icao_idx][2])
            if features == ['lon', 'lat', 'alt']:
                lon_data_for_icao_idx = multitimeseries_data[icao_idx][0]
                lat_data_for_icao_idx = multitimeseries_data[icao_idx][1]
                alt_data_for_icao_idx = multitimeseries_data[icao_idx][2]
            if features == ['lon', 'lat', 'alt', 'hdg']:
                lon_data_for_icao_idx = multitimeseries_data[icao_idx][0]
                lat_data_for_icao_idx = multitimeseries_data[icao_idx][1]
                alt_data_for_icao_idx = multitimeseries_data[icao_idx][2]
                hdg_data_for_icao_idx = multitimeseries_data[icao_idx][3]
            # print('alt_data_for_icao_idx = ', alt_data_for_icao_idx)
            # print('alt_data_for_icao_idx.shape = ', alt_data_for_icao_idx.shape)
            # print('lon_data_for_icao_idx = ', lon_data_for_icao_idx)
            # print('lon_data_for_icao_idx.shape = ', lon_data_for_icao_idx.shape)
            # print('lat_data_for_icao_idx = ', lat_data_for_icao_idx)
            # print('lat_data_for_icao_idx.shape = ', lat_data_for_icao_idx.shape)

            # ax.plot(x_data_array, data[icao_idx], **plt_kwargs, label=str(track_id_array[icao_idx]))
            # ax.scatter(x_data_array, data[icao_idx], **plt_kwargs, label=str(icao_array[icao_idx]))
            # ax.plot(range(0, len(data[icao_idx])), data[icao_idx], alpha=0.2, **plt_kwargs, label=str(track_id_array[icao_idx]))

            # used to print on plot the 'trajectory' legend only for the last trajectory instead of for all
            if icao_idx != cluster_icao_idx[-1]:
                # print('trigger 1')
                # ax.plot(range(0, len(alt_data_for_icao_idx)), alt_data_for_icao_idx, c='gray', alpha=0.3, **plt_kwargs)
                # axs[0, 0].plot(range(0, len(alt_data_for_icao_idx)), alt_data_for_icao_idx, c='gray', alpha=0.3, **plt_kwargs)
                # axs[0, 1].plot(range(0, len(alt_data_for_icao_idx)), lon_data_for_icao_idx, c='gray', alpha=0.3, **plt_kwargs)
                # axs[0, 2].plot(range(0, len(alt_data_for_icao_idx)), lat_data_for_icao_idx, c='gray', alpha=0.3, **plt_kwargs)
                # ax1.plot(range(0, len(alt_data_for_icao_idx)), alt_data_for_icao_idx, c='gray', alpha=opacity, **plt_kwargs)
                # ax2.plot(range(0, len(alt_data_for_icao_idx)), lon_data_for_icao_idx, c='gray', alpha=opacity, **plt_kwargs)
                # ax3.plot(range(0, len(alt_data_for_icao_idx)), lat_data_for_icao_idx, c='gray', alpha=opacity, **plt_kwargs)
                opacity = 0.4
                linewidth = 1
                zorder = 1
                if features == ['lon', 'lat', 'alt']:
                    ax1.plot(range(0, len(lon_data_for_icao_idx)), lon_data_for_icao_idx, c=color_list[cluster], linewidth=linewidth,alpha=opacity, zorder=zorder, **plt_kwargs)
                    ax2.plot(range(0, len(lat_data_for_icao_idx)), lat_data_for_icao_idx, c=color_list[cluster], linewidth=linewidth,alpha=opacity, zorder=zorder, **plt_kwargs)
                    ax3.plot(range(0, len(alt_data_for_icao_idx)), alt_data_for_icao_idx, c=color_list[cluster], linewidth=linewidth,alpha=opacity, zorder=zorder, **plt_kwargs)
                if features == ['lon', 'lat', 'alt', 'hdg']:
                    ax1.plot(range(0, len(lon_data_for_icao_idx)), lon_data_for_icao_idx, c=color_list[cluster], linewidth=linewidth, alpha=opacity, zorder=zorder, **plt_kwargs)
                    ax2.plot(range(0, len(lat_data_for_icao_idx)), lat_data_for_icao_idx, c=color_list[cluster], linewidth=linewidth, alpha=opacity, zorder=zorder, **plt_kwargs)
                    ax3.plot(range(0, len(alt_data_for_icao_idx)), alt_data_for_icao_idx, c=color_list[cluster], linewidth=linewidth, alpha=opacity, zorder=zorder, **plt_kwargs)
                    ax4.plot(range(0, len(hdg_data_for_icao_idx)), hdg_data_for_icao_idx, c=color_list[cluster], linewidth=linewidth, alpha=opacity, zorder=zorder, **plt_kwargs)
            # else:
            #     # print('trigger 2')
            #     # ax.plot(range(0, len(alt_data_for_icao_idx)), alt_data_for_icao_idx, c='gray', alpha=0.3, label='trajectories', **plt_kwargs)
            #     # axs[0, 0].plot(range(0, len(alt_data_for_icao_idx)), alt_data_for_icao_idx, c='gray', alpha=0.3, label='trajectories', **plt_kwargs)
            #     # axs[0, 1].plot(range(0, len(alt_data_for_icao_idx)), lon_data_for_icao_idx, c='gray', alpha=0.3, label='trajectories', **plt_kwargs)
            #     # axs[0, 2].plot(range(0, len(alt_data_for_icao_idx)), lat_data_for_icao_idx, c='gray', alpha=0.3, label='trajectories', **plt_kwargs)
            #     # ax1.plot(range(0, len(alt_data_for_icao_idx)), alt_data_for_icao_idx, c='gray', alpha=opacity, label='cluster_'+str(i), **plt_kwargs)
            #     # ax2.plot(range(0, len(alt_data_for_icao_idx)), lon_data_for_icao_idx, c='gray', alpha=opacity, label='cluster_'+str(i), **plt_kwargs)
            #     # ax3.plot(range(0, len(alt_data_for_icao_idx)), lat_data_for_icao_idx, c='gray', alpha=opacity, label='cluster_'+str(i), **plt_kwargs)
            #     ax1.plot(range(0, len(alt_data_for_icao_idx)), alt_data_for_icao_idx, c=color_list[cluster], alpha=opacity, label='cluster_'+str(i), **plt_kwargs)
            #     ax2.plot(range(0, len(alt_data_for_icao_idx)), lon_data_for_icao_idx, c=color_list[cluster], alpha=opacity, label='cluster_'+str(i), **plt_kwargs)
            #     ax3.plot(range(0, len(alt_data_for_icao_idx)), lat_data_for_icao_idx, c=color_list[cluster], alpha=opacity, label='cluster_'+str(i), **plt_kwargs)
        # ax.plot(range(0, len(alt_data_for_icao_idx)), alt_centroids, c='red', label='centroid', **plt_kwargs)
        # ax1.plot(range(0, len(alt_data_for_icao_idx)), alt_centroids, c='red', linewidth=2,label='centroid', **plt_kwargs)
        # ax2.plot(range(0, len(lon_data_for_icao_idx)), lon_centroids, c='red', linewidth=2, label='centroid', **plt_kwargs)
        # ax3.plot(range(0, len(lat_data_for_icao_idx)), lat_centroids, c='red', linewidth=2, label='centroid', **plt_kwargs)
        linewidth = 4
        zorder = 10
        edgecolor = 'k'
        if features == ['lon', 'lat', 'alt']:
            ax1.plot(range(0, len(lon_data_for_icao_idx)), lon_centroids, c=color_list[cluster], linewidth=linewidth, path_effects=[pe.Stroke(linewidth=linewidth+2, foreground=edgecolor), pe.Normal()], zorder=zorder,label='Centroid ' + str(i), **plt_kwargs)
            ax2.plot(range(0, len(lat_data_for_icao_idx)), lat_centroids, c=color_list[cluster], linewidth=linewidth, path_effects=[pe.Stroke(linewidth=linewidth+2, foreground=edgecolor), pe.Normal()], zorder=zorder,label='Centroid ' + str(i), **plt_kwargs)
            ax3.plot(range(0, len(alt_data_for_icao_idx)), alt_centroids, c=color_list[cluster], linewidth=linewidth, path_effects=[pe.Stroke(linewidth=linewidth+2, foreground=edgecolor), pe.Normal()], zorder=zorder,label='Centroid ' + str(i), **plt_kwargs)
        if features == ['lon', 'lat', 'alt', 'hdg']:
            ax1.plot(range(0, len(lon_data_for_icao_idx)), lon_centroids, c=color_list[cluster], linewidth=linewidth, path_effects=[pe.Stroke(linewidth=linewidth+2, foreground=edgecolor), pe.Normal()], zorder=zorder,label='Centroid '+str(i), **plt_kwargs)
            ax2.plot(range(0, len(lat_data_for_icao_idx)), lat_centroids, c=color_list[cluster], linewidth=linewidth, path_effects=[pe.Stroke(linewidth=linewidth+2, foreground=edgecolor), pe.Normal()], zorder=zorder, label='Centroid '+str(i), **plt_kwargs)
            ax3.plot(range(0, len(alt_data_for_icao_idx)), alt_centroids, c=color_list[cluster], linewidth=linewidth, path_effects=[pe.Stroke(linewidth=linewidth+2, foreground=edgecolor), pe.Normal()], zorder=zorder, label='Centroid '+str(i), **plt_kwargs)
            ax4.plot(range(0, len(hdg_data_for_icao_idx)), hdg_centroids, c=color_list[cluster], linewidth=linewidth, path_effects=[pe.Stroke(linewidth=linewidth+2, foreground=edgecolor), pe.Normal()], zorder=zorder, label='Centroid '+str(i), **plt_kwargs)


    # fig.suptitle('Cluster '+str(cluster_idx)+' out of '+str(cluster))
    fig.suptitle('Number of Clusters = '+str(cluster_idx))
    ax3.legend(loc='upper right')
    # ax3.annotate(str(len(cluster_icao_idx)), xy=(1, 0), xycoords='axes fraction', fontsize=15, horizontalalignment='right', verticalalignment='bottom')

    if features == ['lon', 'lat', 'alt']:
        ax1.title.set_text('Longitude')
        ax2.title.set_text('Latitude')
        ax3.title.set_text('Altitude')
    if features == ['lon', 'lat', 'alt', 'hdg']:
        ax1.title.set_text('Longitude')
        ax2.title.set_text('Latitude')
        ax3.title.set_text('Altitude')
        ax4.title.set_text('Heading')

    if features == ['lon', 'lat', 'alt']:
        ax1.set_ylabel('Normalized Values')
        ax2.set_xlabel('Time Index')
        # ax2.axes.get_yaxis().set_visible(False)
        # ax3.axes.get_yaxis().set_visible(False)
    if features == ['lon', 'lat', 'alt', 'hdg']:
        ax1.set_ylabel('Normalized Values')
        ax2.set_xlabel('Time Index')
        # ax2.axes.get_yaxis().set_visible(False)
        # ax3.axes.get_yaxis().set_visible(False)
        # ax4.axes.get_yaxis().set_visible(False)
    # fig.subplots_adjust(wspace=0, hspace=0)

    fig_name = path + "nb clusters "+str(cluster_idx)+" features vs time.png"
    fig.savefig(fig_name, dpi=200)
    if features == ['lon', 'lat', 'alt']:
        fig.clear(ax1)
        fig.clear(ax2)
        fig.clear(ax3)
    if features == ['lon', 'lat', 'alt', 'hdg']:
        fig.clear(ax1)
        fig.clear(ax2)
        fig.clear(ax3)
        fig.clear(ax4)
    # fig.clear(ax)
    # fig.clear(axs)
    # fig.clear(ax1)
    # fig.clear(ax2)
    # fig.clear(ax3)
    plt.close(fig)
    # ax = None

def multivariate_timeseries_plot3(multitimeseries_data, model_preds, ac_info_data,centroids, color_list, cluster_idx, path, features, ax=None, plt_kwargs={}):
    # temp_data = data[data['icao24'] == ac_id]

    # print('model_preds = ', model_preds)
    # print('model_preds.shape = ', model_preds.shape)

    # print('centroids = ', centroids)
    # print('centroids.shape = ', centroids.shape)
    # print('len(centroids) = ', len(centroids))

    # print('features = ', features)
    # print('len(features) = ', len(features))

    # # make a list of random colors for each unique cluster
    # color_list = get_colors(cluster_idx)
    # color_list = cm.rainbow(np.linspace(0, 1, cluster_idx))
    # color_list = cm.Set2(np.linspace(0, 1, cluster_idx))
    # print('color_list = ', color_list)

    ac_info_data = ac_info_data.astype({'time': 'float', 'lat': 'float', 'lon': 'float', 'geoaltitude': 'float',
                                        'heading': 'float', 'turnrate': 'float', 'acceleration': 'float'})
    # print('ac_info_data =\n', ac_info_data)

    unique_clusters = list(ac_info_data['cluster'].unique())
    unique_clusters = sorted(unique_clusters)
    unique_track_ids = list(ac_info_data['track_id'].unique())


    if features == ['lon', 'lat', 'alt']:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

    if features == ['lon', 'lat', 'alt', 'hdg']:
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)

    # for i, cluster in enumerate(np.unique(model_preds)):
    for i, cluster in enumerate(unique_clusters):
        # print('=================================================')
        print('     --> plotting cluster', str(i),'(from',str(cluster_idx),'total clusters)')
        # print('=================================================')

        # print('cluster name = ', i)
        # cluster_idx = cluster_idx-1
        # cluster_idx = cluster
        # print('cluster     = ', cluster_id)
        # print('cluster     = ', cluster)
        # print('cluster_idx = ', cluster_idx)
        # print('model_preds = ', model_preds)

        # print('centroids   = ', centroids)
        # print('centroids.shape   = ', centroids.shape)
        # print('centroids[cluster_idx]   = ', centroids)
        # print('centroids[cluster_idx].shape   = ', centroids[cluster_idx].shape)
        # alt_centroids = centroids[cluster_idx][0]
        # lon_centroids = centroids[cluster_idx][1]
        # lat_centroids = centroids[cluster_idx][2]

        ac_info_data_for_cluster = ac_info_data[ac_info_data["cluster"] == cluster]
        # print('ac_info_data_for_cluster =\n', ac_info_data_for_cluster)

        if features == ['lon', 'lat', 'alt']:
            lon_centroids = centroids[i][0]
            lat_centroids = centroids[i][1]
            alt_centroids = centroids[i][2]
            # lon_centroids = ac_info_data_for_cluster.groupby(['time'])['lon'].mean().tolist()
            # lat_centroids = ac_info_data_for_cluster.groupby(['time'])['lat'].mean().tolist()
            # alt_centroids = ac_info_data_for_cluster.groupby(['time'])['geoaltitude'].mean().tolist()

        if features == ['lon', 'lat', 'alt', 'hdg']:
            lon_centroids = centroids[i][0]
            lat_centroids = centroids[i][1]
            alt_centroids = centroids[i][2]
            hdg_centroids = centroids[i][3]

        max_range = 0
        for icao_idx, track_id in enumerate(unique_track_ids):
            # print('icao_idx = ', icao_idx)
            # print('track_id = ', track_id)

            lon_data_for_icao_id = ac_info_data_for_cluster[ac_info_data_for_cluster["track_id"] == track_id]
            lat_data_for_icao_id = ac_info_data_for_cluster[ac_info_data_for_cluster["track_id"] == track_id]
            alt_data_for_icao_id = ac_info_data_for_cluster[ac_info_data_for_cluster["track_id"] == track_id]
            # print('lon_data_for_icao_id =\n ', lon_data_for_icao_id)
            # print('lat_data_for_icao_id =\n ', lat_data_for_icao_id)
            # print('alt_data_for_icao_id =\n ', alt_data_for_icao_id)
            # alt_data_for_icao_id = alt_data_for_icao_id.filter(['time', 'heading'], axis=1)
            # print('alt_data_for_icao_id =\n ', alt_data_for_icao_id)
            # alt_data_for_icao_id = list(alt_data_for_icao_id['heading'])
            lon_data_for_icao_id = lon_data_for_icao_id.lon.values.tolist()
            lat_data_for_icao_id = lat_data_for_icao_id.lat.values.tolist()
            alt_data_for_icao_id = alt_data_for_icao_id.geoaltitude.values.tolist()
            # alt_data_for_icao_id = alt_data_for_icao_id['heading'].to_list()
            # print('alt_data_for_icao_id =\n ', alt_data_for_icao_id)
            # print('alt_data_for_icao_id len =\n ', len(alt_data_for_icao_id))

            # print('range(0, len(alt_data_for_icao_id)) =\n ', np.linspace(0, len(alt_data_for_icao_id), 1))
            # print('range list =\n ', range(0, len(alt_data_for_icao_id)))
            # print('range list =\n ', list(range(0, len(alt_data_for_icao_id))))

            if len(alt_data_for_icao_id) > max_range:
                max_range = len(alt_data_for_icao_id)
            # print('max_range = ', max_range)

            # used to print on plot the 'trajectory' legend only for the last trajectory instead of for all
            ax1.plot(list(range(0, len(lon_data_for_icao_id))), lon_data_for_icao_id, c=color_list[i], alpha=0.6,linewidth=1, zorder=1, **plt_kwargs)
            ax2.plot(list(range(0, len(lat_data_for_icao_id))), lat_data_for_icao_id, c=color_list[i], alpha=0.6,linewidth=1, zorder=1, **plt_kwargs)
            ax3.plot(list(range(0, len(alt_data_for_icao_id))), alt_data_for_icao_id, c=color_list[i], alpha=0.6,linewidth=1, zorder=1, **plt_kwargs)

        # lon_centroids_for_cluster = lon_centroids_for_cluster[0:max_range]
        # lat_centroids_for_cluster = lat_centroids_for_cluster[0:max_range]
        # alt_centroids = alt_centroids_for_cluster[0:max_range]
        # alt_centroids_for_cluster = alt_centroids_for_cluster[0:max_range]
        # ax1.plot(list(range(0, len(alt_data_for_icao_id))), alt_centroids_for_cluster, '--', c=color_list[i], linewidth=2,label='centroid_' + str(i), **plt_kwargs)
        # ax1.plot(list(range(0, max_range)), lon_centroids, c=color_list[i], linewidth=4, path_effects=[pe.Stroke(linewidth=6, foreground='k'), pe.Normal()], zorder=10, label='Centroid ' + str(i), **plt_kwargs)
        ax1.plot(list(range(0, len(lon_centroids))), lon_centroids, c=color_list[i], linewidth=4, path_effects=[pe.Stroke(linewidth=6, foreground='k'), pe.Normal()], zorder=10, label='Centroid ' + str(i), **plt_kwargs)
        # ax2.plot(list(range(0, max_range)), lat_centroids, c=color_list[i], linewidth=4, path_effects=[pe.Stroke(linewidth=6, foreground='k'), pe.Normal()], zorder=10, label='Centroid ' + str(i), **plt_kwargs)
        ax2.plot(list(range(0, len(lat_centroids))), lat_centroids, c=color_list[i], linewidth=4, path_effects=[pe.Stroke(linewidth=6, foreground='k'), pe.Normal()], zorder=10, label='Centroid ' + str(i), **plt_kwargs)
        # ax3.plot(list(range(0, max_range)), alt_centroids_for_cluster, c=color_list[i], linewidth=4, path_effects=[pe.Stroke(linewidth=6, foreground='k'), pe.Normal()], zorder=10, label='Centroid ' + str(i), **plt_kwargs)
        # ax3.plot(list(range(0, max_range)), alt_centroids, c=color_list[i], linewidth=4, path_effects=[pe.Stroke(linewidth=6, foreground='k'), pe.Normal()], zorder=10, label='Centroid ' + str(i), **plt_kwargs)
        ax3.plot(list(range(0, len(alt_centroids))), alt_centroids, c=color_list[i], linewidth=4, path_effects=[pe.Stroke(linewidth=6, foreground='k'), pe.Normal()], zorder=10, label='Centroid ' + str(i), **plt_kwargs)

    # fig.suptitle('Cluster '+str(cluster_idx)+' out of '+str(cluster))
    fig.suptitle('Number of Clusters = '+str(cluster_idx))
    ax3.legend(loc='upper right')
    # ax3.annotate(str(len(cluster_icao_idx)), xy=(1, 0), xycoords='axes fraction', fontsize=15, horizontalalignment='right', verticalalignment='bottom')

    if features == ['lon', 'lat', 'alt']:
        ax1.title.set_text('Longitude')
        ax2.title.set_text('Latitude')
        ax3.title.set_text('Altitude')
    if features == ['lon', 'lat', 'alt', 'hdg']:
        ax1.title.set_text('Longitude')
        ax2.title.set_text('Latitude')
        ax3.title.set_text('Altitude')
        ax4.title.set_text('Heading')

    if features == ['lon', 'lat', 'alt']:
        ax1.set_ylabel('Normalized Values')
        ax2.set_xlabel('Time Index')
        # ax2.axes.get_yaxis().set_visible(False)
        # ax3.axes.get_yaxis().set_visible(False)
    if features == ['lon', 'lat', 'alt', 'hdg']:
        ax1.set_ylabel('Normalized Values')
        ax2.set_xlabel('Time Index')
        # ax2.axes.get_yaxis().set_visible(False)
        # ax3.axes.get_yaxis().set_visible(False)
        # ax4.axes.get_yaxis().set_visible(False)
    # fig.subplots_adjust(wspace=0, hspace=0)

    fig_name = path + "nb clusters "+str(cluster_idx)+" features vs time.png"
    fig.savefig(fig_name, dpi=200)
    if features == ['lon', 'lat', 'alt']:
        fig.clear(ax1)
        fig.clear(ax2)
        fig.clear(ax3)
    if features == ['lon', 'lat', 'alt', 'hdg']:
        fig.clear(ax1)
        fig.clear(ax2)
        fig.clear(ax3)
        fig.clear(ax4)
    # fig.clear(ax)
    # fig.clear(axs)
    # fig.clear(ax1)
    # fig.clear(ax2)
    # fig.clear(ax3)
    plt.close(fig)
    # ax = None

def multivariate_timeseries_plot4(multitimeseries_data, model_preds,centroids, color_list, cluster_idx, path, features, ax=None, plt_kwargs={}):
    # temp_data = data[data['icao24'] == ac_id]

    # print('model_preds = ', model_preds)
    # print('model_preds.shape = ', model_preds.shape)

    # print('centroids = ', centroids)
    # print('centroids.shape = ', centroids.shape)
    # print('len(centroids) = ', len(centroids))

    # print('features = ', features)
    # print('len(features) = ', len(features))

    # # make a list of random colors for each unique cluster
    # color_list = get_colors(cluster_idx)
    # color_list = cm.rainbow(np.linspace(0, 1, cluster_idx))
    # color_list = cm.Set2(np.linspace(0, 1, cluster_idx))
    # print('color_list = ', color_list)

    multitimeseries_data = scaler.inverse_transform(multitimeseries_data.reshape(multitimeseries_data.shape[0], -1)).reshape(multitimeseries_data.shape)
    # multitimeseries_data = scaler.inverse_transform(multitimeseries_data)

    centroids = scaler.inverse_transform(centroids.reshape(centroids.shape[0], -1)).reshape(centroids.shape)
    # scaler.inverse_transform(centroids)
    # centroids = scaler.inverse_transform(centroids)

    if features == ['lon', 'lat', 'alt']:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(45, 30))

    if features == ['lon', 'lat', 'alt', 'hdg']:
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(15, 15))

    font_size = 50

    for i, cluster in enumerate(np.unique(model_preds)):
        # print('=================================================')
        print('     --> plotting cluster', str(i),'(from',str(cluster_idx),'total clusters)')
        # print('=================================================')

        # print('cluster name = ', i)
        # cluster_idx = cluster_idx-1
        # cluster_idx = cluster
        # print('cluster     = ', cluster_id)
        # print('cluster     = ', cluster)
        # print('cluster_idx = ', cluster_idx)
        # print('model_preds = ', model_preds)

        # print('centroids   = ', centroids)
        # print('centroids.shape   = ', centroids.shape)
        # print('centroids[cluster_idx]   = ', centroids)
        # print('centroids[cluster_idx].shape   = ', centroids[cluster_idx].shape)
        # alt_centroids = centroids[cluster_idx][0]
        # lon_centroids = centroids[cluster_idx][1]
        # lat_centroids = centroids[cluster_idx][2]

        if features == ['lon', 'lat', 'alt']:
            lon_centroids = centroids[i][0]
            lat_centroids = centroids[i][1]
            alt_centroids = centroids[i][2]
            # lon_centroids = ac_info_data_for_cluster.groupby(['time'])['lon'].mean().tolist()
            # lat_centroids = ac_info_data_for_cluster.groupby(['time'])['lat'].mean().tolist()
            # alt_centroids = ac_info_data_for_cluster.groupby(['time'])['geoaltitude'].mean().tolist()

        if features == ['lon', 'lat', 'alt', 'hdg']:
            lon_centroids = centroids[i][0]
            lat_centroids = centroids[i][1]
            alt_centroids = centroids[i][2]
            hdg_centroids = centroids[i][3]
        # print('alt_centroids = ', alt_centroids)
        # print('lon_centroids = ', lon_centroids)
        # print('lat_centroids = ', lat_centroids)
        # print('alt_centroids.shape = ', alt_centroids.shape)
        # print('lon_centroids.shape = ', lon_centroids.shape)
        # print('lat_centroids.shape = ', lat_centroids.shape)
        # print('     nb in cluster = ', len(lon_centroids))

        # get individual trajectories that are in the cluster i and return their indexes
        # cluster_icao_idx = np.where(model_preds == cluster_idx)[0]
        cluster_icao_idx = np.where(model_preds == i)[0]
        # print('cluster_icao_idx = ', cluster_icao_idx)
        # track_id_array = np.array(track_id_list)
        # print('track_id in cluster = ', track_id_array[np.where(y==cluster_idx)[0]])
        # print('cluster centroid = \n', cluster_centers[cluster_idx])
        # print('nb of traj in cluster', str(i),'= ', len(cluster_icao_idx))


        # if ax is None:
        #     # fig = plt.gcf().set_size_inches(10, 5)
        #     fig = plt.gcf()
        #     # ax = plt.gca()
        #     # fig, axs = plt.subplots(1, 3)
        #     # axs = plt.subplots(1, 3)
        #     fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

        # for icao in icao_array[np.where(y==cluster_idx)[0]]:
        for icao_idx in cluster_icao_idx:
            # print('+++    _idx = ', icao_idx)
            # print('len(cluster_icao_idx) = ', len(cluster_icao_idx))
            # print('cluster_icao_idx[-1] = ', cluster_icao_idx[-1])

            # print('multitimeseries_data[icao_idx] = ', multitimeseries_data[icao_idx])
            # print('multitimeseries_data[icao_idx].shape = ', multitimeseries_data[icao_idx].shape)
            # print('multitimeseries_data[icao_idx][0] = ', multitimeseries_data[icao_idx][0])
            # print('multitimeseries_data[icao_idx][1] = ', multitimeseries_data[icao_idx][1])
            # print('multitimeseries_data[icao_idx][2] = ', multitimeseries_data[icao_idx][2])
            if features == ['lon', 'lat', 'alt']:
                lon_data_for_icao_idx = multitimeseries_data[icao_idx][0]
                lat_data_for_icao_idx = multitimeseries_data[icao_idx][1]
                alt_data_for_icao_idx = multitimeseries_data[icao_idx][2]
            if features == ['lon', 'lat', 'alt', 'hdg']:
                lon_data_for_icao_idx = multitimeseries_data[icao_idx][0]
                lat_data_for_icao_idx = multitimeseries_data[icao_idx][1]
                alt_data_for_icao_idx = multitimeseries_data[icao_idx][2]
                hdg_data_for_icao_idx = multitimeseries_data[icao_idx][3]
            # print('alt_data_for_icao_idx = ', alt_data_for_icao_idx)
            # print('alt_data_for_icao_idx.shape = ', alt_data_for_icao_idx.shape)
            # print('lon_data_for_icao_idx = ', lon_data_for_icao_idx)
            # print('lon_data_for_icao_idx.shape = ', lon_data_for_icao_idx.shape)
            # print('lat_data_for_icao_idx = ', lat_data_for_icao_idx)
            # print('lat_data_for_icao_idx.shape = ', lat_data_for_icao_idx.shape)

            # ax.plot(x_data_array, data[icao_idx], **plt_kwargs, label=str(track_id_array[icao_idx]))
            # ax.scatter(x_data_array, data[icao_idx], **plt_kwargs, label=str(icao_array[icao_idx]))
            # ax.plot(range(0, len(data[icao_idx])), data[icao_idx], alpha=0.2, **plt_kwargs, label=str(track_id_array[icao_idx]))

            # used to print on plot the 'trajectory' legend only for the last trajectory instead of for all
            if icao_idx != cluster_icao_idx[-1]:
                # print('trigger 1')
                # ax.plot(range(0, len(alt_data_for_icao_idx)), alt_data_for_icao_idx, c='gray', alpha=0.3, **plt_kwargs)
                # axs[0, 0].plot(range(0, len(alt_data_for_icao_idx)), alt_data_for_icao_idx, c='gray', alpha=0.3, **plt_kwargs)
                # axs[0, 1].plot(range(0, len(alt_data_for_icao_idx)), lon_data_for_icao_idx, c='gray', alpha=0.3, **plt_kwargs)
                # axs[0, 2].plot(range(0, len(alt_data_for_icao_idx)), lat_data_for_icao_idx, c='gray', alpha=0.3, **plt_kwargs)
                # ax1.plot(range(0, len(alt_data_for_icao_idx)), alt_data_for_icao_idx, c='gray', alpha=opacity, **plt_kwargs)
                # ax2.plot(range(0, len(alt_data_for_icao_idx)), lon_data_for_icao_idx, c='gray', alpha=opacity, **plt_kwargs)
                # ax3.plot(range(0, len(alt_data_for_icao_idx)), lat_data_for_icao_idx, c='gray', alpha=opacity, **plt_kwargs)
                opacity = 0.4
                linewidth = 1
                zorder = 1
                if features == ['lon', 'lat', 'alt']:
                    ax1.plot(range(0, len(lon_data_for_icao_idx)), lon_data_for_icao_idx, c=color_list[cluster], linewidth=linewidth,alpha=opacity, zorder=zorder, **plt_kwargs)
                    ax2.plot(range(0, len(lat_data_for_icao_idx)), lat_data_for_icao_idx, c=color_list[cluster], linewidth=linewidth,alpha=opacity, zorder=zorder, **plt_kwargs)
                    ax3.plot(range(0, len(alt_data_for_icao_idx)), alt_data_for_icao_idx, c=color_list[cluster], linewidth=linewidth,alpha=opacity, zorder=zorder, **plt_kwargs)
                if features == ['lon', 'lat', 'alt', 'hdg']:
                    ax1.plot(range(0, len(lon_data_for_icao_idx)), lon_data_for_icao_idx, c=color_list[cluster], linewidth=linewidth, alpha=opacity, zorder=zorder, **plt_kwargs)
                    ax2.plot(range(0, len(lat_data_for_icao_idx)), lat_data_for_icao_idx, c=color_list[cluster], linewidth=linewidth, alpha=opacity, zorder=zorder, **plt_kwargs)
                    ax3.plot(range(0, len(alt_data_for_icao_idx)), alt_data_for_icao_idx, c=color_list[cluster], linewidth=linewidth, alpha=opacity, zorder=zorder, **plt_kwargs)
                    ax4.plot(range(0, len(hdg_data_for_icao_idx)), hdg_data_for_icao_idx, c=color_list[cluster], linewidth=linewidth, alpha=opacity, zorder=zorder, **plt_kwargs)
            # else:
            #     # print('trigger 2')
            #     # ax.plot(range(0, len(alt_data_for_icao_idx)), alt_data_for_icao_idx, c='gray', alpha=0.3, label='trajectories', **plt_kwargs)
            #     # axs[0, 0].plot(range(0, len(alt_data_for_icao_idx)), alt_data_for_icao_idx, c='gray', alpha=0.3, label='trajectories', **plt_kwargs)
            #     # axs[0, 1].plot(range(0, len(alt_data_for_icao_idx)), lon_data_for_icao_idx, c='gray', alpha=0.3, label='trajectories', **plt_kwargs)
            #     # axs[0, 2].plot(range(0, len(alt_data_for_icao_idx)), lat_data_for_icao_idx, c='gray', alpha=0.3, label='trajectories', **plt_kwargs)
            #     # ax1.plot(range(0, len(alt_data_for_icao_idx)), alt_data_for_icao_idx, c='gray', alpha=opacity, label='cluster_'+str(i), **plt_kwargs)
            #     # ax2.plot(range(0, len(alt_data_for_icao_idx)), lon_data_for_icao_idx, c='gray', alpha=opacity, label='cluster_'+str(i), **plt_kwargs)
            #     # ax3.plot(range(0, len(alt_data_for_icao_idx)), lat_data_for_icao_idx, c='gray', alpha=opacity, label='cluster_'+str(i), **plt_kwargs)
            #     ax1.plot(range(0, len(alt_data_for_icao_idx)), alt_data_for_icao_idx, c=color_list[cluster], alpha=opacity, label='cluster_'+str(i), **plt_kwargs)
            #     ax2.plot(range(0, len(alt_data_for_icao_idx)), lon_data_for_icao_idx, c=color_list[cluster], alpha=opacity, label='cluster_'+str(i), **plt_kwargs)
            #     ax3.plot(range(0, len(alt_data_for_icao_idx)), lat_data_for_icao_idx, c=color_list[cluster], alpha=opacity, label='cluster_'+str(i), **plt_kwargs)
        # ax.plot(range(0, len(alt_data_for_icao_idx)), alt_centroids, c='red', label='centroid', **plt_kwargs)
        # ax1.plot(range(0, len(alt_data_for_icao_idx)), alt_centroids, c='red', linewidth=2,label='centroid', **plt_kwargs)
        # ax2.plot(range(0, len(lon_data_for_icao_idx)), lon_centroids, c='red', linewidth=2, label='centroid', **plt_kwargs)
        # ax3.plot(range(0, len(lat_data_for_icao_idx)), lat_centroids, c='red', linewidth=2, label='centroid', **plt_kwargs)
        linewidth = 10
        zorder = 10
        edgecolor = 'k'
        if features == ['lon', 'lat', 'alt']:
            ax1.plot(range(0, len(lon_data_for_icao_idx)), lon_centroids, c=color_list[cluster], linewidth=linewidth, path_effects=[pe.Stroke(linewidth=linewidth+2, foreground=edgecolor), pe.Normal()], zorder=zorder,label='Centroid of Cluster ' + str(i), **plt_kwargs)
            ax2.plot(range(0, len(lat_data_for_icao_idx)), lat_centroids, c=color_list[cluster], linewidth=linewidth, path_effects=[pe.Stroke(linewidth=linewidth+2, foreground=edgecolor), pe.Normal()], zorder=zorder,label='Centroid of Cluster ' + str(i), **plt_kwargs)
            ax3.plot(range(0, len(alt_data_for_icao_idx)), alt_centroids, c=color_list[cluster], linewidth=linewidth, path_effects=[pe.Stroke(linewidth=linewidth+2, foreground=edgecolor), pe.Normal()], zorder=zorder,label='Centroid of Cluster ' + str(i), **plt_kwargs)
        if features == ['lon', 'lat', 'alt', 'hdg']:
            ax1.plot(range(0, len(lon_data_for_icao_idx)), lon_centroids, c=color_list[cluster], linewidth=linewidth, path_effects=[pe.Stroke(linewidth=linewidth+2, foreground=edgecolor), pe.Normal()], zorder=zorder,label='Centroid of Cluster '+str(i), **plt_kwargs)
            ax2.plot(range(0, len(lat_data_for_icao_idx)), lat_centroids, c=color_list[cluster], linewidth=linewidth, path_effects=[pe.Stroke(linewidth=linewidth+2, foreground=edgecolor), pe.Normal()], zorder=zorder, label='Centroid of Cluster '+str(i), **plt_kwargs)
            ax3.plot(range(0, len(alt_data_for_icao_idx)), alt_centroids, c=color_list[cluster], linewidth=linewidth, path_effects=[pe.Stroke(linewidth=linewidth+2, foreground=edgecolor), pe.Normal()], zorder=zorder, label='Centroid of Cluster '+str(i), **plt_kwargs)
            ax4.plot(range(0, len(hdg_data_for_icao_idx)), hdg_centroids, c=color_list[cluster], linewidth=linewidth, path_effects=[pe.Stroke(linewidth=linewidth+2, foreground=edgecolor), pe.Normal()], zorder=zorder, label='Centroid of Cluster '+str(i), **plt_kwargs)


    # fig.suptitle('Cluster '+str(cluster_idx)+' out of '+str(cluster))
    fig.suptitle('Number of Clusters = '+str(cluster_idx), fontsize=font_size)
    ax3.legend(loc='upper right', fontsize=font_size).set_zorder(100)
    # ax3.annotate(str(len(cluster_icao_idx)), xy=(1, 0), xycoords='axes fraction', fontsize=15, horizontalalignment='right', verticalalignment='bottom')

    ax1.tick_params(axis="x", labelsize=font_size)
    ax2.tick_params(axis="x", labelsize=font_size)
    ax3.tick_params(axis="x", labelsize=font_size)

    ax1.tick_params(axis="y", labelsize=font_size)
    ax2.tick_params(axis="y", labelsize=font_size)
    ax3.tick_params(axis="y", labelsize=font_size)

    if features == ['lon', 'lat', 'alt']:
        # ax1.title.set_text('Longitude [deg]', fontsize=font_size)
        # ax2.title.set_text('Latitude [deg]', fontsize=font_size)
        # ax3.title.set_text('Altitude [m]', fontsize=font_size)
        ax1.set_title('Longitude [deg]', fontsize=font_size)
        ax2.set_title('Latitude [deg]', fontsize=font_size)
        ax3.set_title('Altitude [m]', fontsize=font_size)
    if features == ['lon', 'lat', 'alt', 'hdg']:
        ax1.title.set_text('Longitude', fontsize=font_size)
        ax2.title.set_text('Latitude', fontsize=font_size)
        ax3.title.set_text('Altitude', fontsize=font_size)
        ax4.title.set_text('Heading', fontsize=font_size)

    if features == ['lon', 'lat', 'alt']:
        # ax1.set_ylabel('Normalized Values')
        ax1.set_xlabel('Time Index', fontsize=font_size)
        ax2.set_xlabel('Time Index', fontsize=font_size)
        ax3.set_xlabel('Time Index', fontsize=font_size)
        # ax2.axes.get_yaxis().set_visible(False)
        # ax3.axes.get_yaxis().set_visible(False)
    if features == ['lon', 'lat', 'alt', 'hdg']:
        ax1.set_ylabel('Normalized Values', fontsize=font_size)
        ax1.set_xlabel('Time Index', fontsize=font_size)
        ax2.set_xlabel('Time Index', fontsize=font_size)
        ax3.set_xlabel('Time Index', fontsize=font_size)
        # ax2.axes.get_yaxis().set_visible(False)
        # ax3.axes.get_yaxis().set_visible(False)
        # ax4.axes.get_yaxis().set_visible(False)
    # fig.subplots_adjust(wspace=0, hspace=0)
    plt.show()

    fig_name = path + "nb clusters "+str(cluster_idx)+" features vs time.png"
    fig.savefig(fig_name, dpi=200)
    if features == ['lon', 'lat', 'alt']:
        fig.clear(ax1)
        fig.clear(ax2)
        fig.clear(ax3)
    if features == ['lon', 'lat', 'alt', 'hdg']:
        fig.clear(ax1)
        fig.clear(ax2)
        fig.clear(ax3)
        fig.clear(ax4)
    # fig.clear(ax)
    # fig.clear(axs)
    # fig.clear(ax1)
    # fig.clear(ax2)
    # fig.clear(ax3)
    plt.close(fig)
    # ax = None

def ac_info_n_cluster_data_merger(ac_info_data, cluster_track_id_df):
    # print('ac_info_data =\n', ac_info_data)
    # print('cluster_track_id_df =\n', cluster_track_id_df)

    unique_track_ids_list = []
    for col in cluster_track_id_df.columns:
        temp = list(cluster_track_id_df[col].unique())
        # print('temp = \n', temp)
        unique_track_ids_list = unique_track_ids_list + temp
    # print('unique_track_ids_list = \n', unique_track_ids_list)

    ac_info_data['cluster'] = 0
    # ac_info_df.loc[ac_info_df.track_id == 'AA9A0A_1', 'cluster'] = "cluster_0"
    for unique_track in unique_track_ids_list:
        for col in cluster_track_id_df.columns:
            if (unique_track in cluster_track_id_df[col].unique()) == True:
                # print('---------------')
                # print(unique_track+' is in '+str(col))
                ac_info_data.loc[ac_info_data.track_id == unique_track, 'cluster'] = str(col)

    ac_info_df = ac_info_data[ac_info_data.cluster != 0]
    # print('ac_info_data =\n', ac_info_data)

    # drop all the ac traj that are not used in the kmeans model
    ac_info_data = ac_info_data[ac_info_data.cluster != 0]

    return ac_info_data

import pyproj
geodesic = pyproj.Geod(ellps='WGS84')
def get_bearing_from_2pts(lon1, lat1, lon2, lat2):
    fwd_azimuth, back_azimuth, distance = geodesic.inv(lon1, lat1, lon2, lat2)
    # print('-----')
    # print('fwd_azimuth = ', fwd_azimuth)
    if fwd_azimuth < 0:
        fwd_azimuth = abs(fwd_azimuth)+180
    # print('fwd_azimuth = ', fwd_azimuth)
    return fwd_azimuth

# def NdArrayFct(ndarray):
#
#     ndarray
#
#     return new_ndarray

# old version using average backup
# def heading_timeseries_plot_old(features, multitimeseries_data, ac_info_data, model_preds, centroids, cluster_idx, colors, path, ax=None, plt_kwargs={}):
#     # temp_data = data[data['icao24'] == ac_id]
#     # print('\n-----------------------')
#     # print('heading plot')
#     # print('-----------------------')
#
#
#     # print('multitimeseries_data =\n', multitimeseries_data)
#     multitimeseries_data = scaler.inverse_transform(multitimeseries_data.reshape(multitimeseries_data.shape[0], -1)).reshape(multitimeseries_data.shape)
#     # print('multitimeseries_data =\n', multitimeseries_data)
#
#     # print('len(multitimeseries_data) =', len(multitimeseries_data))
#     # print('multitimeseries_data.shape =', multitimeseries_data.shape)
#
#     # print('multitimeseries_data =\n', multitimeseries_data)
#     # print('multitimeseries_data[0] =\n', multitimeseries_data[0]) # traj 1 data
#     #
#     # print('multitimeseries_data[0][0] =\n', multitimeseries_data[0][0]) # traj 1 lon
#     # print('multitimeseries_data[0][1] =\n', multitimeseries_data[0][1]) # traj 1 lat
#     # print('multitimeseries_data[0][2] =\n', multitimeseries_data[0][2]) # traj 1 alt
#     #
#     # print('multitimeseries_data[0][0][0] =\n', multitimeseries_data[0][0][0]) # traj 1 lon value of time 1
#     # print('multitimeseries_data[0][0][1] =\n', multitimeseries_data[0][0][1]) # traj 1 lat value of time 1
#     # print('multitimeseries_data[0][0][1] =\n', multitimeseries_data[0][0][2]) # traj 1 alt value of time 1
#     #
#     # print('multitimeseries_data[1][0][2] =\n', multitimeseries_data[1][0][2]) # traj 2 lon value of time 2
#     # print('multitimeseries_data[1][1][2] =\n', multitimeseries_data[1][1][2]) # traj 2 lat value of time 2
#     # print('multitimeseries_data[1][2][2] =\n', multitimeseries_data[1][2][2]) # traj 2 alt value of time 2
#
#     track_id = 0
#     heading_data = np.asarray([[]])
#     heading_data = np.zeros(150)
#     for lon, lat, alt in multitimeseries_data:
#         # print('-----------------')
#         # print('track_id = ', track_id)
#         # print('lon = ', lon)
#         # print('lat = ', lat)
#         # print('alt = ', alt)
#
#         # print('len(lon) = ', len(lon)) # len of lon = total time (150)
#         # print('len(lat) = ', len(lat))
#         # print('len(alt) = ', len(alt))
#
#         headings_for_track_id = []
#         for time_id in range(0, len(lon)):
#             # if time_id > 0:
#             # print('===')
#             # print('time_id = ', time_id)
#             # print('lon[time_id] = ', round(lon[time_id], 8))
#             # print('lat[time_id] = ', round(lat[time_id], 8))
#             # print('alt[time_id] = ', alt[time_id])
#
#             # print('lon[time_id-1] = ', round(lon[time_id-1], 8))
#             # print('lat[time_id-1] = ', round(lat[time_id-1], 8))
#             # print('alt[time_id-1] = ', alt[time_id-1])
#             # print('headings_for_track_id = \n', headings_for_track_id)
#
#             if time_id == 0:
#                 heading_at_time_id = get_bearing_from_2pts(lon1=lon[time_id-1], lat1=lat[time_id-1], lon2=lon[time_id], lat2=lat[time_id])
#             else:
#                 if (lon[time_id] == lon[time_id-1]) & (lat[time_id] == lat[time_id-1]):
#                     # print('!no position change')
#                     heading_at_time_id = headings_for_track_id[-1]
#                 else:
#                     heading_at_time_id = get_bearing_from_2pts(lon1=lon[time_id-1], lat1=lat[time_id-1], lon2=lon[time_id], lat2=lat[time_id])
#                     # print('heading_at_time_id = ', heading_at_time_id)
#                     # heading_at_time_id = get_bearing_from_2pts(lon1=lon[time_id], lat1=lat[time_id], lon2=lon[time_id-1], lat2=lat[time_id-1])
#                     # print('heading_at_time_id = ', heading_at_time_id)
#
#             # headings_for_track_id.append([1, 2])
#             headings_for_track_id.append(round(heading_at_time_id, 2))
#             # print('headings_for_track_id = \n', headings_for_track_id)
#             # if time_id == 2:
#             #     break
#         headings_for_track_id = np.asarray(headings_for_track_id)
#         # print('headings_for_track_id = \n', headings_for_track_id)
#         # print('len headings_for_track_id   =  ', len(headings_for_track_id))
#         # print('type headings_for_track_id  =  ', type(headings_for_track_id))
#         # print('headings_for_track_id.shape =  ', headings_for_track_id.shape)
#         # new_multitimeseries_data[track_id] = np.append(headings_for_track_id[:-1], len(new_multitimeseries_data[track_id]))
#         # new_multitimeseries_data[track_id] = np.append(arr=new_multitimeseries_data[track_id], values=headings_for_track_id[:-1], axis=0)
#         # print('new_multitimeseries_data[track_id] =\n', new_multitimeseries_data[track_id])
#         # print('len new_multitimeseries_data[track_id]   = ', len(new_multitimeseries_data[track_id]))
#         # print('type new_multitimeseries_data[track_id]  = ', type(new_multitimeseries_data[track_id]))
#         # print('new_multitimeseries_data[track_id].shape = ', new_multitimeseries_data[track_id].shape)
#         # new_multitimeseries_data[track_id] = np.append(headings_for_track_id[:-1], new_multitimeseries_data[track_id])
#         # new_multitimeseries_data[track_id] = new_multitimeseries_data[track_id].insert(1, headings_for_track_id)
#         # new_multitimeseries_data[track_id] = np.insert(new_multitimeseries_data[track_id], 1, 1)
#         # new_multitimeseries_data[track_id] = np.insert(new_multitimeseries_data[track_id], 1, 1)
#         # heading_data[track_id] = heading_data.append(headings_for_track_id, axis=1)
#         # heading_data[track_id] = headings_for_track_id[:-1]
#         # heading_data = np.vstack((heading_data, headings_for_track_id[:-1]))
#         heading_data = np.vstack((heading_data, headings_for_track_id))
#         # heading_data = np.append(headings_for_track_id[:-1], heading_data, axis=0)
#         # print('heading_data =\n', heading_data)
#         # print('len heading_data   = ', len(heading_data))
#         # print('type heading_data  = ', type(heading_data))
#         # print('heading_data.shape = ', heading_data.shape)
#
#         # if track_id == 2:
#         #     break
#         track_id += 1
#
#     # multitimeseries_data = np.append(new_multitimeseries_data, len(multitimeseries_data))
#     # print('multitimeseries_data =\n', multitimeseries_data)
#     # print('multitimeseries_data.shape =\n', multitimeseries_data.shape)
#     # print('heading_data =\n', heading_data)
#     heading_data = heading_data[1:]
#     # print('heading_data =\n', heading_data)
#     # print('len heading_data   = ', len(heading_data))
#     # print('type heading_data  = ', type(heading_data))
#     # print('heading_data.shape = ', heading_data.shape)
#
#
#     print('centroids =\n', centroids)
#     # centroids = scaler.inverse_transform(centroids.reshape(centroids.shape[0], -1)).reshape(centroids.shape)
#     # print('centroids =\n', centroids)
#     print('len(centroids)  = ', len(centroids))
#     print('type(centroids) = ', type(centroids))
#     print('centroids.shape = ', centroids.shape)
#
#     track_id = 0
#     centroid_heading_data = np.zeros(150)
#     for cluster_id in range(0, len(centroids)):
#         print('-----------------')
#         print('cluster_id = ', cluster_id)
#         print('centroids[cluster_id] = \n', centroids[cluster_id])
#
#         lon = centroids[cluster_id][0]
#         lat = centroids[cluster_id][1]
#         alt = centroids[cluster_id][2]
#
#         print('lon = ', lon[:5])
#         print('lat = ', lat[:5])
#         print('alt = ', alt[:5])
#
#         headings_for_track_id = []
#         for time_id in range(0, len(lon)):
#             # if time_id > 0:
#             # print('===')
#             # print('time_id = ', time_id)
#             # print('lon[time_id] = ', round(lon[time_id], 8))
#             # print('lat[time_id] = ', round(lat[time_id], 8))
#             # print('alt[time_id] = ', alt[time_id])
#
#             # print('lon[time_id-1] = ', round(lon[time_id-1], 8))
#             # print('lat[time_id-1] = ', round(lat[time_id-1], 8))
#             # print('alt[time_id-1] = ', alt[time_id-1])
#             # print('headings_for_track_id = \n', headings_for_track_id)
#
#             if time_id == 0:
#                 heading_at_time_id = get_bearing_from_2pts(lon1=lon[time_id - 1], lat1=lat[time_id - 1], lon2=lon[time_id],
#                                                            lat2=lat[time_id])
#             else:
#                 if (lon[time_id] == lon[time_id - 1]) & (lat[time_id] == lat[time_id - 1]):
#                     # print('!no position change')
#                     heading_at_time_id = headings_for_track_id[-1]
#                 else:
#                     heading_at_time_id = get_bearing_from_2pts(lon1=lon[time_id - 1], lat1=lat[time_id - 1],
#                                                                lon2=lon[time_id], lat2=lat[time_id])
#                     # print('heading_at_time_id = ', heading_at_time_id)
#                     # heading_at_time_id = get_bearing_from_2pts(lon1=lon[time_id], lat1=lat[time_id], lon2=lon[time_id-1], lat2=lat[time_id-1])
#                     # print('heading_at_time_id = ', heading_at_time_id)
#
#             # headings_for_track_id.append([1, 2])
#             headings_for_track_id.append(round(heading_at_time_id, 2))
#             # print('headings_for_track_id = \n', headings_for_track_id)
#             # if time_id == 2:
#             #     break
#         headings_for_track_id = np.asarray(headings_for_track_id)
#         centroid_heading_data = np.vstack((centroid_heading_data, headings_for_track_id))
#     centroid_heading_data = heading_data[1:]
#
#     print('centroid_heading_data =\n', centroid_heading_data)
#     print('len(centroid_heading_data)  = ', len(centroid_heading_data))
#     print('type(centroid_heading_data) = ', type(centroid_heading_data))
#     print('centroid_heading_data.shape = ', centroid_heading_data.shape)
#
#
#     # print('ac_info_data =\n', ac_info_data)
#     ac_info_data = ac_info_data.astype({'time': 'float', 'lat': 'float', 'lon': 'float', 'geoaltitude': 'float',
#                                         'heading': 'float', 'turnrate': 'float', 'acceleration': 'float'})
#
#     # ac_info_data = ac_info_data.head(max_time_index)
#     # print('ac_info_data.dtypes =\n', ac_info_data.dtypes)
#
#     # print('model_preds = ', model_preds)
#     # print('model_preds.shape = ', model_preds.shape)
#
#     # print('centroids = ', centroids)
#     # print('centroids.shape = ', centroids.shape)
#     # print('len(centroids) = ', len(centroids))
#
#     # print('features = ', features)
#     # print('len(features) = ', len(features))
#
#     unique_clusters = list(ac_info_data['cluster'].unique())
#     unique_clusters = sorted(unique_clusters)
#     unique_track_ids = list(ac_info_data['track_id'].unique())
#     # print('unique_clusters =\n', unique_clusters)
#     # print('unique_track_ids =\n', unique_track_ids)
#
#     fig, ax1 = plt.subplots(1, 1)
#
#     for i, cluster in enumerate(unique_clusters):
#         # print('=================================================')
#         print('     --> plotting cluster', str(i), '(from', str(cluster_idx), 'total clusters)')
#         # print('=================================================')
#
#         ac_info_data_for_cluster = ac_info_data[ac_info_data["cluster"] == cluster]
#
#         hdg_centroids_for_cluster = ac_info_data_for_cluster.groupby(['time'])['heading'].mean().tolist()
#
#         # print('i = ', i)
#         # print('cluster     = ', cluster)
#         # print('cluster_idx = ', cluster_idx)
#         # print('ac_info_data_for_cluster = \n', ac_info_data_for_cluster)
#         # print('hdg_centroids_for_cluster = \n', hdg_centroids_for_cluster)
#
#         max_range = 0
#         for icao_idx, track_id in enumerate(unique_track_ids):
#             # print('icao_idx = ', icao_idx)
#             # print('track_id = ', track_id)
#
#             hdg_data_for_icao_id = ac_info_data_for_cluster[ac_info_data_for_cluster["track_id"] == track_id]
#             # print('hdg_data_for_icao_id =\n ', hdg_data_for_icao_id)
#             # hdg_data_for_icao_id = hdg_data_for_icao_id.filter(['time', 'heading'], axis=1)
#             # print('hdg_data_for_icao_id =\n ', hdg_data_for_icao_id)
#             # hdg_data_for_icao_id = list(hdg_data_for_icao_id['heading'])
#             hdg_data_for_icao_id = hdg_data_for_icao_id.heading.values.tolist()
#             # hdg_data_for_icao_id = hdg_data_for_icao_id['heading'].to_list()
#             # print('hdg_data_for_icao_id =\n ', hdg_data_for_icao_id)
#             # print('hdg_data_for_icao_id len =\n ', len(hdg_data_for_icao_id))
#
#             # print('range(0, len(hdg_data_for_icao_id)) =\n ', np.linspace(0, len(hdg_data_for_icao_id), 1))
#             # print('range list =\n ', range(0, len(hdg_data_for_icao_id)))
#             # print('range list =\n ', list(range(0, len(hdg_data_for_icao_id))))
#
#             if len(hdg_data_for_icao_id) > max_range:
#                 max_range = len(hdg_data_for_icao_id)
#             # print('max_range = ', max_range)
#
#             # used to print on plot the 'trajectory' legend only for the last trajectory instead of for all
#             ax1.plot(list(range(0, len(hdg_data_for_icao_id))), hdg_data_for_icao_id, c=color_list[i], alpha=0.6, linewidth=1, zorder=1, **plt_kwargs)
#
#         hdg_centroids_for_cluster = hdg_centroids_for_cluster[0:max_range]
#         # ax1.plot(list(range(0, len(hdg_data_for_icao_id))), hdg_centroids_for_cluster, '--', c=color_list[i], linewidth=2,label='centroid_' + str(i), **plt_kwargs)
#         # ax1.plot(list(range(0, max_range)), hdg_centroids_for_cluster, '--', c=color_list[i], linewidth=4, path_effects=[pe.Stroke(linewidth=6, foreground='k'), pe.Normal()],label='centroid_' + str(i), **plt_kwargs)
#         ax1.plot(list(range(0, max_range)), hdg_centroids_for_cluster, c=color_list[i], linewidth=4, path_effects=[pe.Stroke(linewidth=6, foreground='k'), pe.Normal()], zorder=10,label='Average of Cluster ' + str(i), **plt_kwargs)
#
#     # fig.suptitle('Cluster '+str(cluster_idx)+' out of '+str(cluster))
#     fig.suptitle('Number of Clusters = ' + str(cluster_idx))
#     ax1.legend(loc='upper right').set_zorder(100)
#
#     # ax3.annotate(str(len(cluster_icao_idx)), xy=(1, 0), xycoords='axes fraction', fontsize=15,
#     #              horizontalalignment='right', verticalalignment='bottom')
#
#     # plt.xlim([0, len(hdg_data_for_icao_id)])
#     plt.xlim([0, max_time_index])
#
#     # if features == ['lon', 'lat', 'alt', 'hdg']:
#     ax1.title.set_text('Heading')
#     # ax2.title.set_text('Latitude')
#     # ax3.title.set_text('Altitude')
#     # ax4.title.set_text('Heading')
#
#     # if features == ['lon', 'lat', 'alt', 'hdg']:
#     ax1.set_ylabel('Heading [deg]')
#     ax1.set_xlabel('Time Index')
#     # ax2.axes.get_yaxis().set_visible(False)
#     fig.subplots_adjust(wspace=0, hspace=0)
#
#     fig_name = path + "nb clusters " + str(len(centroids)) + " hdg vs time plot.png"
#     fig.savefig(fig_name, dpi=200)
#
#     # if features == ['lon', 'lat', 'alt', 'hdg']:
#     fig.clear(ax1)
#     # fig.clear(ax2)
#     # fig.clear(ax3)
#     plt.close(fig)
#     # ax = None


def heading_timeseries_plot(features, multitimeseries_data, ac_info_data, model_preds, centroids, cluster_idx, colors, path, ax=None, plt_kwargs={}):
    # temp_data = data[data['icao24'] == ac_id]
    # print('\n-----------------------')
    # print('heading plot')
    # print('-----------------------')


    # print('multitimeseries_data =\n', multitimeseries_data)
    multitimeseries_data = scaler.inverse_transform(multitimeseries_data.reshape(multitimeseries_data.shape[0], -1)).reshape(multitimeseries_data.shape)
    # print('multitimeseries_data =\n', multitimeseries_data)

    # print('len(multitimeseries_data) =', len(multitimeseries_data))
    # print('multitimeseries_data.shape =', multitimeseries_data.shape)

    # print('multitimeseries_data =\n', multitimeseries_data)
    # print('multitimeseries_data[0] =\n', multitimeseries_data[0]) # traj 1 data
    #
    # print('multitimeseries_data[0][0] =\n', multitimeseries_data[0][0]) # traj 1 lon
    # print('multitimeseries_data[0][1] =\n', multitimeseries_data[0][1]) # traj 1 lat
    # print('multitimeseries_data[0][2] =\n', multitimeseries_data[0][2]) # traj 1 alt
    #
    # print('multitimeseries_data[0][0][0] =\n', multitimeseries_data[0][0][0]) # traj 1 lon value of time 1
    # print('multitimeseries_data[0][0][1] =\n', multitimeseries_data[0][0][1]) # traj 1 lat value of time 1
    # print('multitimeseries_data[0][0][1] =\n', multitimeseries_data[0][0][2]) # traj 1 alt value of time 1
    #
    # print('multitimeseries_data[1][0][2] =\n', multitimeseries_data[1][0][2]) # traj 2 lon value of time 2
    # print('multitimeseries_data[1][1][2] =\n', multitimeseries_data[1][1][2]) # traj 2 lat value of time 2
    # print('multitimeseries_data[1][2][2] =\n', multitimeseries_data[1][2][2]) # traj 2 alt value of time 2

    track_id = 0
    heading_data = np.asarray([[]])
    heading_data = np.zeros(150)
    for lon, lat, alt in multitimeseries_data:
        # print('-----------------')
        # print('track_id = ', track_id)
        # print('lon = ', lon)
        # print('lat = ', lat)
        # print('alt = ', alt)

        # print('len(lon) = ', len(lon)) # len of lon = total time (150)
        # print('len(lat) = ', len(lat))
        # print('len(alt) = ', len(alt))

        headings_for_track_id = []
        for time_id in range(0, len(lon)):
            # if time_id > 0:
            # print('===')
            # print('time_id = ', time_id)
            # print('lon[time_id] = ', round(lon[time_id], 8))
            # print('lat[time_id] = ', round(lat[time_id], 8))
            # print('alt[time_id] = ', alt[time_id])

            # print('lon[time_id-1] = ', round(lon[time_id-1], 8))
            # print('lat[time_id-1] = ', round(lat[time_id-1], 8))
            # print('alt[time_id-1] = ', alt[time_id-1])
            # print('headings_for_track_id = \n', headings_for_track_id)

            if time_id == 0:
                heading_at_time_id = get_bearing_from_2pts(lon1=lon[time_id-1], lat1=lat[time_id-1], lon2=lon[time_id], lat2=lat[time_id])
            else:
                if (lon[time_id] == lon[time_id-1]) & (lat[time_id] == lat[time_id-1]):
                    # print('!no position change')
                    heading_at_time_id = headings_for_track_id[-1]
                else:
                    heading_at_time_id = get_bearing_from_2pts(lon1=lon[time_id-1], lat1=lat[time_id-1], lon2=lon[time_id], lat2=lat[time_id])
                    # print('heading_at_time_id = ', heading_at_time_id)
                    # heading_at_time_id = get_bearing_from_2pts(lon1=lon[time_id], lat1=lat[time_id], lon2=lon[time_id-1], lat2=lat[time_id-1])
                    # print('heading_at_time_id = ', heading_at_time_id)

            # headings_for_track_id.append([1, 2])
            headings_for_track_id.append(round(heading_at_time_id, 2))
            # print('headings_for_track_id = \n', headings_for_track_id)
            # if time_id == 2:
            #     break
        headings_for_track_id = np.asarray(headings_for_track_id)
        # print('headings_for_track_id = \n', headings_for_track_id)
        # print('len headings_for_track_id   =  ', len(headings_for_track_id))
        # print('type headings_for_track_id  =  ', type(headings_for_track_id))
        # print('headings_for_track_id.shape =  ', headings_for_track_id.shape)
        # new_multitimeseries_data[track_id] = np.append(headings_for_track_id[:-1], len(new_multitimeseries_data[track_id]))
        # new_multitimeseries_data[track_id] = np.append(arr=new_multitimeseries_data[track_id], values=headings_for_track_id[:-1], axis=0)
        # print('new_multitimeseries_data[track_id] =\n', new_multitimeseries_data[track_id])
        # print('len new_multitimeseries_data[track_id]   = ', len(new_multitimeseries_data[track_id]))
        # print('type new_multitimeseries_data[track_id]  = ', type(new_multitimeseries_data[track_id]))
        # print('new_multitimeseries_data[track_id].shape = ', new_multitimeseries_data[track_id].shape)
        # new_multitimeseries_data[track_id] = np.append(headings_for_track_id[:-1], new_multitimeseries_data[track_id])
        # new_multitimeseries_data[track_id] = new_multitimeseries_data[track_id].insert(1, headings_for_track_id)
        # new_multitimeseries_data[track_id] = np.insert(new_multitimeseries_data[track_id], 1, 1)
        # new_multitimeseries_data[track_id] = np.insert(new_multitimeseries_data[track_id], 1, 1)
        # heading_data[track_id] = heading_data.append(headings_for_track_id, axis=1)
        # heading_data[track_id] = headings_for_track_id[:-1]
        # heading_data = np.vstack((heading_data, headings_for_track_id[:-1]))
        heading_data = np.vstack((heading_data, headings_for_track_id))
        # heading_data = np.append(headings_for_track_id[:-1], heading_data, axis=0)
        # print('heading_data =\n', heading_data)
        # print('len heading_data   = ', len(heading_data))
        # print('type heading_data  = ', type(heading_data))
        # print('heading_data.shape = ', heading_data.shape)

        # if track_id == 2:
        #     break
        track_id += 1

    # multitimeseries_data = np.append(new_multitimeseries_data, len(multitimeseries_data))
    # print('multitimeseries_data =\n', multitimeseries_data)
    # print('multitimeseries_data.shape =\n', multitimeseries_data.shape)
    # print('heading_data =\n', heading_data)
    heading_data = heading_data[1:]
    # print('heading_data =\n', heading_data)
    # print('len heading_data   = ', len(heading_data))
    # print('type heading_data  = ', type(heading_data))
    # print('heading_data.shape = ', heading_data.shape)


    # print('centroids =\n', centroids)
    # centroids = scaler.inverse_transform(centroids.reshape(centroids.shape[0], -1)).reshape(centroids.shape)
    # print('centroids =\n', centroids)
    # print('len(centroids)  = ', len(centroids))
    # print('type(centroids) = ', type(centroids))
    # print('centroids.shape = ', centroids.shape)

    track_id = 0
    centroid_heading_data = np.zeros(150)
    for cluster_id in range(0, len(centroids)):
        # print('-----------------')
        # print('cluster_id = ', cluster_id)
        # print('centroids[cluster_id] = \n', centroids[cluster_id])

        lon = centroids[cluster_id][0]
        lat = centroids[cluster_id][1]
        alt = centroids[cluster_id][2]

        # print('lon = ', lon[:5])
        # print('lat = ', lat[:5])
        # print('alt = ', alt[:5])

        headings_for_track_id = []
        for time_id in range(0, len(lon)):
            # if time_id > 0:
            # print('===')
            # print('time_id = ', time_id)
            # print('lon[time_id] = ', round(lon[time_id], 8))
            # print('lat[time_id] = ', round(lat[time_id], 8))
            # print('alt[time_id] = ', alt[time_id])

            # print('lon[time_id-1] = ', round(lon[time_id-1], 8))
            # print('lat[time_id-1] = ', round(lat[time_id-1], 8))
            # print('alt[time_id-1] = ', alt[time_id-1])
            # print('headings_for_track_id = \n', headings_for_track_id)

            if time_id == 0:
                heading_at_time_id = get_bearing_from_2pts(lon1=lon[time_id - 1], lat1=lat[time_id - 1], lon2=lon[time_id],
                                                           lat2=lat[time_id])
            else:
                if (lon[time_id] == lon[time_id - 1]) & (lat[time_id] == lat[time_id - 1]):
                    # print('!no position change')
                    heading_at_time_id = headings_for_track_id[-1]
                else:
                    heading_at_time_id = get_bearing_from_2pts(lon1=lon[time_id - 1], lat1=lat[time_id - 1],
                                                               lon2=lon[time_id], lat2=lat[time_id])
                    # print('heading_at_time_id = ', heading_at_time_id)
                    # heading_at_time_id = get_bearing_from_2pts(lon1=lon[time_id], lat1=lat[time_id], lon2=lon[time_id-1], lat2=lat[time_id-1])
                    # print('heading_at_time_id = ', heading_at_time_id)

            # headings_for_track_id.append([1, 2])
            headings_for_track_id.append(round(heading_at_time_id, 2))
            # print('headings_for_track_id = \n', headings_for_track_id)
            # if time_id == 2:
            #     break
        headings_for_track_id = np.asarray(headings_for_track_id)
        centroid_heading_data = np.vstack((centroid_heading_data, headings_for_track_id))
    centroid_heading_data = centroid_heading_data[1:]

    # print('centroid_heading_data =\n', centroid_heading_data)
    # print('len(centroid_heading_data)  = ', len(centroid_heading_data))
    # print('type(centroid_heading_data) = ', type(centroid_heading_data))
    # print('centroid_heading_data.shape = ', centroid_heading_data.shape)


    # print('ac_info_data =\n', ac_info_data)
    ac_info_data = ac_info_data.astype({'time': 'float', 'lat': 'float', 'lon': 'float', 'geoaltitude': 'float',
                                        'heading': 'float', 'turnrate': 'float', 'acceleration': 'float'})

    # ac_info_data = ac_info_data.head(max_time_index)
    # print('ac_info_data.dtypes =\n', ac_info_data.dtypes)

    # print('model_preds = ', model_preds)
    # print('model_preds.shape = ', model_preds.shape)

    # print('centroids = ', centroids)
    # print('centroids.shape = ', centroids.shape)
    # print('len(centroids) = ', len(centroids))

    # print('features = ', features)
    # print('len(features) = ', len(features))

    unique_clusters = list(ac_info_data['cluster'].unique())
    unique_clusters = sorted(unique_clusters)
    unique_track_ids = list(ac_info_data['track_id'].unique())
    # print('unique_clusters =\n', unique_clusters)
    # print('unique_track_ids =\n', unique_track_ids)

    fig, ax1 = plt.subplots(1, 1)

    # for i, cluster in enumerate(unique_clusters):
    for i, cluster in enumerate(np.unique(model_preds)):
        # print('=================================================')
        print('     --> plotting cluster', str(i), '(from', str(cluster_idx), 'total clusters)')
        # print('=================================================')
        hdg_centroids_for_cluster = centroid_heading_data[i]
        # print('hdg_centroids_for_cluster =\n', hdg_centroids_for_cluster)

        cluster_icao_idx = np.where(model_preds == i)[0]
        for icao_idx in cluster_icao_idx:
            # print('===')
            # print('icao_idx = ', icao_idx)
            hdg_data_for_icao_id = heading_data[icao_idx]
            # print('hdg_data_for_icao_id = \n', hdg_data_for_icao_id)

            # used to print on plot the 'trajectory' legend only for the last trajectory instead of for all
            ax1.plot(list(range(0, len(hdg_data_for_icao_id))), hdg_data_for_icao_id, c=color_list[i], alpha=0.6, linewidth=1, zorder=1, **plt_kwargs)

        # hdg_centroids_for_cluster = hdg_centroids_for_cluster[0:max_range]
        # ax1.plot(list(range(0, len(hdg_data_for_icao_id))), hdg_centroids_for_cluster, '--', c=color_list[i], linewidth=2,label='centroid_' + str(i), **plt_kwargs)
        # ax1.plot(list(range(0, max_range)), hdg_centroids_for_cluster, '--', c=color_list[i], linewidth=4, path_effects=[pe.Stroke(linewidth=6, foreground='k'), pe.Normal()],label='centroid_' + str(i), **plt_kwargs)
        # ax1.plot(list(range(0, max_range)), hdg_centroids_for_cluster, c=color_list[i], linewidth=4, path_effects=[pe.Stroke(linewidth=6, foreground='k'), pe.Normal()], zorder=10,label='Average of Cluster ' + str(i), **plt_kwargs)
        ax1.plot(list(range(0, len(hdg_centroids_for_cluster))), hdg_centroids_for_cluster, c=color_list[i], linewidth=4, path_effects=[pe.Stroke(linewidth=6, foreground='k'), pe.Normal()], zorder=10,label='Centroid of Cluster ' + str(i), **plt_kwargs)

    # fig.suptitle('Cluster '+str(cluster_idx)+' out of '+str(cluster))
    fig.suptitle('Number of Clusters = ' + str(cluster_idx))
    ax1.legend(loc='upper right').set_zorder(100)

    # ax3.annotate(str(len(cluster_icao_idx)), xy=(1, 0), xycoords='axes fraction', fontsize=15,
    #              horizontalalignment='right', verticalalignment='bottom')

    # plt.xlim([0, len(hdg_data_for_icao_id)])
    plt.xlim([0, max_time_index])

    # if features == ['lon', 'lat', 'alt', 'hdg']:
    ax1.title.set_text('Heading')
    # ax2.title.set_text('Latitude')
    # ax3.title.set_text('Altitude')
    # ax4.title.set_text('Heading')

    # if features == ['lon', 'lat', 'alt', 'hdg']:
    ax1.set_ylabel('Heading [deg]')
    ax1.set_xlabel('Time Index')
    # ax2.axes.get_yaxis().set_visible(False)
    fig.subplots_adjust(wspace=0, hspace=0)

    fig_name = path + "nb clusters " + str(len(centroids)) + " hdg vs time plot.png"
    fig.savefig(fig_name, dpi=200)

    # if features == ['lon', 'lat', 'alt', 'hdg']:
    fig.clear(ax1)
    # fig.clear(ax2)
    # fig.clear(ax3)
    plt.close(fig)
    # ax = None

def altitude_timeseries_plot_old(features, multitimeseries_data, ac_info_data, model_preds, centroids, cluster_idx, colors,path, ax=None, plt_kwargs={}):
    # temp_data = data[data['icao24'] == ac_id]
    # print('\n-----------------------')
    # print('heading plot')
    # print('-----------------------')

    # print('ac_info_data =\n', ac_info_data)

    ac_info_data = ac_info_data.astype({'time': 'float', 'lat': 'float', 'lon': 'float', 'geoaltitude': 'float',
                                        'heading': 'float', 'turnrate': 'float', 'acceleration': 'float'})
    # ac_info_data = ac_info_data.head(max_time_index)

    # print('ac_info_data.dtypes =\n', ac_info_data.dtypes)

    # print('model_preds = ', model_preds)
    # print('model_preds.shape = ', model_preds.shape)

    # print('centroids = ', centroids)
    # print('centroids.shape = ', centroids.shape)
    # print('len(centroids) = ', len(centroids))

    # print('features = ', features)
    # print('len(features) = ', len(features))

    unique_clusters = list(ac_info_data['cluster'].unique())
    unique_clusters = sorted(unique_clusters)
    unique_track_ids = list(ac_info_data['track_id'].unique())
    # print('unique_clusters =\n', unique_clusters)
    # print('unique_track_ids =\n', unique_track_ids)

    fig, ax1 = plt.subplots(1, 1)

    for i, cluster in enumerate(unique_clusters):
        # print('=================================================')
        print('     --> plotting cluster', str(i), '(from', str(cluster_idx), 'total clusters)')
        # print('=================================================')

        ac_info_data_for_cluster = ac_info_data[ac_info_data["cluster"] == cluster]

        alt_centroids_for_cluster = ac_info_data_for_cluster.groupby(['time'])['geoaltitude'].mean().tolist()

        # print('i = ', i)
        # print('cluster     = ', cluster)
        # print('cluster_idx = ', cluster_idx)
        # print('ac_info_data_for_cluster = \n', ac_info_data_for_cluster)
        # print('alt_centroids_for_cluster = \n', alt_centroids_for_cluster)

        max_range = 0
        for icao_idx, track_id in enumerate(unique_track_ids):
            # print('icao_idx = ', icao_idx)
            # print('track_id = ', track_id)

            alt_data_for_icao_id = ac_info_data_for_cluster[ac_info_data_for_cluster["track_id"] == track_id]
            # print('alt_data_for_icao_id =\n ', alt_data_for_icao_id)
            # alt_data_for_icao_id = alt_data_for_icao_id.filter(['time', 'heading'], axis=1)
            # print('alt_data_for_icao_id =\n ', alt_data_for_icao_id)
            # alt_data_for_icao_id = list(alt_data_for_icao_id['heading'])
            alt_data_for_icao_id = alt_data_for_icao_id.geoaltitude.values.tolist()
            # alt_data_for_icao_id = alt_data_for_icao_id['heading'].to_list()
            # print('alt_data_for_icao_id =\n ', alt_data_for_icao_id)
            # print('alt_data_for_icao_id len =\n ', len(alt_data_for_icao_id))

            # print('range(0, len(alt_data_for_icao_id)) =\n ', np.linspace(0, len(alt_data_for_icao_id), 1))
            # print('range list =\n ', range(0, len(alt_data_for_icao_id)))
            # print('range list =\n ', list(range(0, len(alt_data_for_icao_id))))

            if len(alt_data_for_icao_id) > max_range:
                max_range = len(alt_data_for_icao_id)
            # print('max_range = ', max_range)

            # used to print on plot the 'trajectory' legend only for the last trajectory instead of for all
            ax1.plot(list(range(0, len(alt_data_for_icao_id))), alt_data_for_icao_id, c=color_list[i], alpha=0.6,linewidth=1, zorder=1, **plt_kwargs)

        alt_centroids_for_cluster = alt_centroids_for_cluster[0:max_range]
        # ax1.plot(list(range(0, len(alt_data_for_icao_id))), alt_centroids_for_cluster, '--', c=color_list[i], linewidth=2,label='centroid_' + str(i), **plt_kwargs)
        ax1.plot(list(range(0, max_range)), alt_centroids_for_cluster, c=color_list[i], linewidth=4, path_effects=[pe.Stroke(linewidth=6, foreground='k'), pe.Normal()], zorder=10, label='centroid_' + str(i), **plt_kwargs)

    # fig.suptitle('Cluster '+str(cluster_idx)+' out of '+str(cluster))
    fig.suptitle('Number of Clusters = ' + str(cluster_idx))
    # ax3.legend(loc='upper right')
    # ax3.annotate(str(len(cluster_icao_idx)), xy=(1, 0), xycoords='axes fraction', fontsize=15,
    #              horizontalalignment='right', verticalalignment='bottom')

    # plt.xlim([0, len(alt_data_for_icao_id)])
    plt.xlim([0, max_time_index])

    # if features == ['lon', 'lat', 'alt', 'hdg']:
    ax1.title.set_text('Altitude')
    # ax2.title.set_text('Latitude')
    # ax3.title.set_text('Altitude')
    # ax4.title.set_text('Heading')

    # if features == ['lon', 'lat', 'alt', 'hdg']:
    ax1.set_ylabel('Altitude [m]')
    ax1.set_xlabel('Time Index')
    # ax2.axes.get_yaxis().set_visible(False)
    fig.subplots_adjust(wspace=0, hspace=0)

    fig_name = path + "nb clusters " + str(len(centroids)) + " alt vs time plot.png"
    fig.savefig(fig_name, dpi=200)

    # if features == ['lon', 'lat', 'alt', 'hdg']:
    fig.clear(ax1)
    # fig.clear(ax2)
    # fig.clear(ax3)
    plt.close(fig)
    # ax = None

def altitude_timeseries_plot(features, multitimeseries_data, ac_info_data, model_preds, centroids, cluster_idx, colors, path, ax=None, plt_kwargs={}):
    # temp_data = data[data['icao24'] == ac_id]
    # print('\n-----------------------')
    # print('heading plot')
    # print('-----------------------')


    # print('multitimeseries_data =\n', multitimeseries_data)
    multitimeseries_data = scaler.inverse_transform(multitimeseries_data.reshape(multitimeseries_data.shape[0], -1)).reshape(multitimeseries_data.shape)
    # print('multitimeseries_data =\n', multitimeseries_data)

    # print('len(multitimeseries_data) =', len(multitimeseries_data))
    # print('multitimeseries_data.shape =', multitimeseries_data.shape)

    # print('multitimeseries_data =\n', multitimeseries_data)
    # print('multitimeseries_data[0] =\n', multitimeseries_data[0]) # traj 1 data
    #
    # print('multitimeseries_data[0][0] =\n', multitimeseries_data[0][0]) # traj 1 lon
    # print('multitimeseries_data[0][1] =\n', multitimeseries_data[0][1]) # traj 1 lat
    # print('multitimeseries_data[0][2] =\n', multitimeseries_data[0][2]) # traj 1 alt
    #
    # print('multitimeseries_data[0][0][0] =\n', multitimeseries_data[0][0][0]) # traj 1 lon value of time 1
    # print('multitimeseries_data[0][0][1] =\n', multitimeseries_data[0][0][1]) # traj 1 lat value of time 1
    # print('multitimeseries_data[0][0][1] =\n', multitimeseries_data[0][0][2]) # traj 1 alt value of time 1
    #
    # print('multitimeseries_data[1][0][2] =\n', multitimeseries_data[1][0][2]) # traj 2 lon value of time 2
    # print('multitimeseries_data[1][1][2] =\n', multitimeseries_data[1][1][2]) # traj 2 lat value of time 2
    # print('multitimeseries_data[1][2][2] =\n', multitimeseries_data[1][2][2]) # traj 2 alt value of time 2

    track_id = 0
    alt_data = np.asarray([[]])
    alt_data = np.zeros(150)
    for lon, lat, alt in multitimeseries_data:
        # print('-----------------')
        # print('track_id = ', track_id)
        # print('lon = ', lon)
        # print('lat = ', lat)
        # print('alt = ', alt)

        # print('len(lon) = ', len(lon)) # len of lon = total time (150)
        # print('len(lat) = ', len(lat))
        # print('len(alt) = ', len(alt))

        alt_for_track_id = []
        for time_id in range(0, len(lon)):
            alt_for_track_id.append(round(alt[time_id], 4))
        alt_for_track_id = np.asarray(alt_for_track_id)
        alt_data = np.vstack((alt_data, alt_for_track_id))
        track_id += 1

    # multitimeseries_data = np.append(new_multitimeseries_data, len(multitimeseries_data))
    # print('multitimeseries_data =\n', multitimeseries_data)
    # print('multitimeseries_data.shape =\n', multitimeseries_data.shape)
    # print('heading_data =\n', heading_data)
    # heading_data = heading_data[1:]
    # print('heading_data =\n', heading_data)
    # print('len heading_data   = ', len(heading_data))
    # print('type heading_data  = ', type(heading_data))
    # print('heading_data.shape = ', heading_data.shape)


    # print('centroids =\n', centroids)
    # centroids = scaler.inverse_transform(centroids.reshape(centroids.shape[0], -1)).reshape(centroids.shape)
    # print('centroids =\n', centroids)
    # print('len(centroids)  = ', len(centroids))
    # print('type(centroids) = ', type(centroids))
    # print('centroids.shape = ', centroids.shape)

    track_id = 0
    centroid_alt_data = np.zeros(150)
    for cluster_id in range(0, len(centroids)):
        # print('-----------------')
        # print('cluster_id = ', cluster_id)
        # print('centroids[cluster_id] = \n', centroids[cluster_id])

        lon = centroids[cluster_id][0]
        lat = centroids[cluster_id][1]
        alt = centroids[cluster_id][2]

        # print('lon = ', lon[:5])
        # print('lat = ', lat[:5])
        # print('alt = ', alt[:5])

        alt_for_track_id = []
        for time_id in range(0, len(alt)):
            alt_for_track_id.append(round(alt[time_id], 4))
        alt_for_track_id = np.asarray(alt_for_track_id)
        centroid_alt_data = np.vstack((centroid_alt_data, alt_for_track_id))
    # centroid_heading_data = centroid_heading_data[1:]

    # print('centroid_heading_data =\n', centroid_heading_data)
    # print('len(centroid_heading_data)  = ', len(centroid_heading_data))
    # print('type(centroid_heading_data) = ', type(centroid_heading_data))
    # print('centroid_heading_data.shape = ', centroid_heading_data.shape)


    # print('ac_info_data =\n', ac_info_data)
    ac_info_data = ac_info_data.astype({'time': 'float', 'lat': 'float', 'lon': 'float', 'geoaltitude': 'float',
                                        'heading': 'float', 'turnrate': 'float', 'acceleration': 'float'})

    # ac_info_data = ac_info_data.head(max_time_index)
    # print('ac_info_data.dtypes =\n', ac_info_data.dtypes)

    # print('model_preds = ', model_preds)
    # print('model_preds.shape = ', model_preds.shape)

    # print('centroids = ', centroids)
    # print('centroids.shape = ', centroids.shape)
    # print('len(centroids) = ', len(centroids))

    # print('features = ', features)
    # print('len(features) = ', len(features))

    unique_clusters = list(ac_info_data['cluster'].unique())
    unique_clusters = sorted(unique_clusters)
    unique_track_ids = list(ac_info_data['track_id'].unique())
    # print('unique_clusters =\n', unique_clusters)
    # print('unique_track_ids =\n', unique_track_ids)

    fig, ax1 = plt.subplots(1, 1)

    # for i, cluster in enumerate(unique_clusters):
    for i, cluster in enumerate(np.unique(model_preds)):
        # print('=================================================')
        print('     --> plotting cluster', str(i), '(from', str(cluster_idx), 'total clusters)')
        # print('=================================================')
        alt_centroids_for_cluster = centroid_alt_data[i]
        # print('hdg_centroids_for_cluster =\n', hdg_centroids_for_cluster)

        cluster_icao_idx = np.where(model_preds == i)[0]
        for icao_idx in cluster_icao_idx:
            # print('===')
            # print('icao_idx = ', icao_idx)
            alt_data_for_icao_id = alt_data[icao_idx]
            # print('hdg_data_for_icao_id = \n', hdg_data_for_icao_id)

            # used to print on plot the 'trajectory' legend only for the last trajectory instead of for all
            ax1.plot(list(range(0, len(alt_data_for_icao_id))), alt_data_for_icao_id, c=color_list[i], alpha=0.6, linewidth=1, zorder=1, **plt_kwargs)

        # hdg_centroids_for_cluster = hdg_centroids_for_cluster[0:max_range]
        # ax1.plot(list(range(0, len(hdg_data_for_icao_id))), hdg_centroids_for_cluster, '--', c=color_list[i], linewidth=2,label='centroid_' + str(i), **plt_kwargs)
        # ax1.plot(list(range(0, max_range)), hdg_centroids_for_cluster, '--', c=color_list[i], linewidth=4, path_effects=[pe.Stroke(linewidth=6, foreground='k'), pe.Normal()],label='centroid_' + str(i), **plt_kwargs)
        # ax1.plot(list(range(0, max_range)), hdg_centroids_for_cluster, c=color_list[i], linewidth=4, path_effects=[pe.Stroke(linewidth=6, foreground='k'), pe.Normal()], zorder=10,label='Average of Cluster ' + str(i), **plt_kwargs)
        ax1.plot(list(range(0, len(alt_centroids_for_cluster))), alt_centroids_for_cluster, c=color_list[i], linewidth=4, path_effects=[pe.Stroke(linewidth=6, foreground='k'), pe.Normal()], zorder=10,label='Centroid of Cluster ' + str(i), **plt_kwargs)

    # fig.suptitle('Cluster '+str(cluster_idx)+' out of '+str(cluster))
    fig.suptitle('Number of Clusters = ' + str(cluster_idx))
    ax1.legend(loc='upper right').set_zorder(100)

    # ax3.annotate(str(len(cluster_icao_idx)), xy=(1, 0), xycoords='axes fraction', fontsize=15,
    #              horizontalalignment='right', verticalalignment='bottom')

    # plt.xlim([0, len(hdg_data_for_icao_id)])
    plt.xlim([0, max_time_index])

    # if features == ['lon', 'lat', 'alt', 'hdg']:
    ax1.title.set_text('Altitude')
    # ax2.title.set_text('Latitude')
    # ax3.title.set_text('Altitude')
    # ax4.title.set_text('Heading')

    # if features == ['lon', 'lat', 'alt', 'hdg']:
    ax1.set_ylabel('Altitude [m]')
    ax1.set_xlabel('Time Index')
    # ax2.axes.get_yaxis().set_visible(False)
    fig.subplots_adjust(wspace=0, hspace=0)

    fig_name = path + "nb clusters " + str(len(centroids)) + " alt vs time plot.png"
    fig.savefig(fig_name, dpi=200)

    # if features == ['lon', 'lat', 'alt', 'hdg']:
    fig.clear(ax1)
    # fig.clear(ax2)
    # fig.clear(ax3)
    plt.close(fig)
    # ax = None

def silhouette_plot(multitimeseries_data, model_preds, centroids, color_list, sil_vals, cluster_idx, path, ax=None, plt_kwargs={}):
    # temp_data = data[data['icao24'] == ac_id]

    cluster_idx = cluster_idx-1
    # print('cluster_idx = ', cluster_idx)
    # print('model_preds = ', model_preds)
    # print('sil_vals = ', sil_vals)

    # print('centroids   = ', centroids)
    # print('centroids.shape   = ', centroids.shape)
    # print('centroids[cluster_idx]   = ', centroids)
    # print('centroids[cluster_idx].shape   = ', centroids[cluster_idx].shape)
    # alt_centroids = centroids[cluster_idx][0]
    # lon_centroids = centroids[cluster_idx][1]
    # lat_centroids = centroids[cluster_idx][2]
    # print('alt_centroids = ', alt_centroids)
    # print('lon_centroids = ', lon_centroids)
    # print('lat_centroids = ', lat_centroids)
    # print('alt_centroids.shape = ', alt_centroids.shape)
    # print('lon_centroids.shape = ', lon_centroids.shape)
    # print('lat_centroids.shape = ', lat_centroids.shape)

    # cluster_icao_idx = np.where(model_preds==cluster_idx)[0]
    # print('cluster_icao_idx = ', cluster_icao_idx)
    # track_id_array = np.array(track_id_list)
    # print('track_id in cluster = ', track_id_array[np.where(y==cluster_idx)[0]])
    # print('cluster centroid = \n', cluster_centers[cluster_idx])
    # print('nb of traj in cluster = ', len(cluster_icao_idx))
    # color_list = cm.Set2(np.linspace(0, 1, cluster_idx))

    font_size = 15
    if ax is None:
        # fig = plt.gcf().set_size_inches(10, 5)
        fig = plt.gcf()
        ax = plt.gca()

    # Build the Silhouette plot
    y_ticks = []
    y_lower, y_upper = 0, 0
    for i, cluster in enumerate(np.unique(model_preds)):
        # print('++++++++++++++++++')
        # print('i =', i)
        # print('cluster =', cluster)
        # print('color_list[cluster] =', color_list[cluster])

        cluster_silhouette_vals = sil_vals[model_preds == cluster]
        avg_cluster_silhouette_vals = np.average(cluster_silhouette_vals)
        cluster_silhouette_vals.sort()
        y_upper += len(cluster_silhouette_vals)
        # ax.barh(range(y_lower, y_upper), cluster_silhouette_vals, edgecolor='none', height=1)
        ax.barh(range(y_lower, y_upper), cluster_silhouette_vals, color=color_list[cluster], edgecolor='none', height=1)
        ax.text(-0.03, (y_lower + y_upper) / 2, str(i + 1), fontsize=font_size)
        ax.annotate(str(round(avg_cluster_silhouette_vals, 2)), xy=((avg_cluster_silhouette_vals+0.05*avg_cluster_silhouette_vals)/2, (y_lower + y_upper) / 2), xycoords='data', fontsize=font_size, horizontalalignment='left', verticalalignment='bottom')
        y_lower += len(cluster_silhouette_vals)

    # Get the average silhouette score and plot it as a vertical line
    avg_score = np.mean(sil_vals)
    # ax.axvline(avg_score, linestyle='--', linewidth=2, color='green')
    # ax.set_xticklabels(fontsize=font_size)
    ax.tick_params(axis="x", labelsize=font_size)
    ax.set_xlim([-0.1, 1])
    ax.set_yticks([])
    ax.set_xlabel('Silhouette coefficient values', fontsize=font_size)
    ax.set_ylabel('Cluster labels', fontsize=font_size)
    ax.set_title('Silhouette plot for the various clusters', y=1.02, fontsize=font_size);
    # ax.annotate(str(len(cluster_icao_idx)), xy=(1, 0), xycoords='axes fraction', fontsize=15, horizontalalignment='right', verticalalignment='bottom')
    # ax.annotate(str(avg_score), xy=(avg_score, 0), xycoords='axes fraction', fontsize=15, horizontalalignment='right', verticalalignment='bottom')
    # ax.annotate(str(round(avg_score, 2)), xy=(avg_score, 0), xycoords='axes points', fontsize=15, horizontalalignment='left', verticalalignment='bottom')
    # ax.annotate(str(round(avg_score, 2)), xy=(avg_score+0.05*avg_score, -0.1), xycoords='data', fontsize=15, horizontalalignment='left', verticalalignment='bottom')

    # ax.plot(range(0, len(alt_data_for_icao_idx)), alt_centroids, c='red', label='centroid', **plt_kwargs)
    # ax.legend(loc='upper right')
    # ax.annotate(str(len(cluster_icao_idx)), xy=(1, 0), xycoords='axes fraction', fontsize=15, horizontalalignment='right', verticalalignment='bottom')
    # fig.suptitle('Cluster '+str(cluster_idx))
    # ax.set_xlabel('Time')
    # ax.set_ylabel('Altitude [m]')
    fig_name = path + "nb clusters "+str(cluster+1)+" silhouette plot"+".png"
    fig.savefig(fig_name, dpi=200)
    fig.clear(ax)
    plt.close(fig)
    # return ax

def multivariate_timeseries_plot_nosubplots(multitimeseries_data, model_preds, centroids, cluster_idx, path, ax=None, plt_kwargs={}):
    # temp_data = data[data['icao24'] == ac_id]

    cluster_idx = cluster_idx-1
    print('cluster_idx = ', cluster_idx)
    print('model_preds = ', model_preds)


    # print('centroids   = ', centroids)
    # print('centroids.shape   = ', centroids.shape)
    # print('centroids[cluster_idx]   = ', centroids)
    print('centroids[cluster_idx].shape   = ', centroids[cluster_idx].shape)
    alt_centroids = centroids[cluster_idx][0]
    lon_centroids = centroids[cluster_idx][1]
    lat_centroids = centroids[cluster_idx][2]
    # print('alt_centroids = ', alt_centroids)
    # print('lon_centroids = ', lon_centroids)
    # print('lat_centroids = ', lat_centroids)
    # print('alt_centroids.shape = ', alt_centroids.shape)
    # print('lon_centroids.shape = ', lon_centroids.shape)
    # print('lat_centroids.shape = ', lat_centroids.shape)

    cluster_icao_idx = np.where(model_preds==cluster_idx)[0]
    # print('cluster_icao_idx = ', cluster_icao_idx)
    # track_id_array = np.array(track_id_list)
    # print('track_id in cluster = ', track_id_array[np.where(y==cluster_idx)[0]])
    # print('cluster centroid = \n', cluster_centers[cluster_idx])
    print('nb of traj in cluster = ', len(cluster_icao_idx))

    if ax is None:
        # fig = plt.gcf().set_size_inches(10, 5)
        fig = plt.gcf()
        ax = plt.gca()
    # for icao in icao_array[np.where(y==cluster_idx)[0]]:
    for icao_idx in cluster_icao_idx:
        # print('======================================')
        # print('icao_idx = ', icao_idx)
        # print('len(cluster_icao_idx) = ', len(cluster_icao_idx))
        # print('cluster_icao_idx[-1] = ', cluster_icao_idx[-1])

        # print('multitimeseries_data[icao_idx] = ', multitimeseries_data[icao_idx])
        # print('multitimeseries_data[icao_idx].shape = ', multitimeseries_data[icao_idx].shape)
        # print('multitimeseries_data[icao_idx][0] = ', multitimeseries_data[icao_idx][0])
        # print('multitimeseries_data[icao_idx][1] = ', multitimeseries_data[icao_idx][1])
        # print('multitimeseries_data[icao_idx][2] = ', multitimeseries_data[icao_idx][2])

        alt_data_for_icao_idx = multitimeseries_data[icao_idx][0]
        lon_data_for_icao_idx = multitimeseries_data[icao_idx][1]
        lat_data_for_icao_idx = multitimeseries_data[icao_idx][2]
        # print('alt_data_for_icao_idx = ', alt_data_for_icao_idx)
        # print('alt_data_for_icao_idx.shape = ', alt_data_for_icao_idx.shape)
        # print('lon_data_for_icao_idx = ', lon_data_for_icao_idx)
        # print('lon_data_for_icao_idx.shape = ', lon_data_for_icao_idx.shape)
        # print('lat_data_for_icao_idx = ', lat_data_for_icao_idx)
        # print('lat_data_for_icao_idx.shape = ', lat_data_for_icao_idx.shape)

        # ax.plot(x_data_array, data[icao_idx], **plt_kwargs, label=str(track_id_array[icao_idx]))
        # ax.scatter(x_data_array, data[icao_idx], **plt_kwargs, label=str(icao_array[icao_idx]))
        # ax.plot(range(0, len(data[icao_idx])), data[icao_idx], alpha=0.2, **plt_kwargs, label=str(track_id_array[icao_idx]))

        # used to print on plot the 'trajectory' legend only for the last trajectory instead of for all
        if icao_idx != cluster_icao_idx[-1]:
            # print('trigger 1')
            ax.plot(range(0, len(alt_data_for_icao_idx)), alt_data_for_icao_idx, c='gray', alpha=0.3, **plt_kwargs)
        else:
            # print('trigger 2')
            ax.plot(range(0, len(alt_data_for_icao_idx)), alt_data_for_icao_idx, c='gray', alpha=0.3, label='trajectories', **plt_kwargs)
    ax.plot(range(0, len(alt_data_for_icao_idx)), alt_centroids, c='red', label='centroid', **plt_kwargs)
    ax.legend(loc='upper right')
    ax.annotate(str(len(cluster_icao_idx)), xy=(1, 0), xycoords='axes fraction', fontsize=15, horizontalalignment='right', verticalalignment='bottom')
    fig.suptitle('Cluster '+str(cluster_idx))
    ax.set_xlabel('Time')
    ax.set_ylabel('Altitude [m]')
    fig_name = path + "alt_cluster_" + str(cluster_idx) + "_of_" + str(n_clusters) + ".png"
    fig.savefig(fig_name, dpi=200)
    fig.clear(ax)
    plt.close(fig)
    # return ax

def twod_traj_plot(ac_info_df, multitimeseries_data, model_preds, centroids, cluster_idx, path, ax=None, plt_kwargs={}):
    # temp_data = data[data['icao24'] == ac_id]

    cluster_idx = cluster_idx-1
    print('cluster_idx = ', cluster_idx)
    print('model_preds = ', model_preds)

    hover_data_list = [ac_info_df['icao24'],
                       ac_info_df['local time'],
                       # ac_info_df['geoaltitude'],
                       # ac_info_df['velocity'],
                       # ac_info_df['heading'],
                       # ac_info_df['vertrate'],
                       ac_info_df['TYPE AIRCRAFT'],
                       ac_info_df['CERTIFICATION'],
                       ac_info_df['TYPE ENGINE'],
                       ac_info_df['NO-ENG'],
                       ac_info_df['NO-SEATS'],
                       ac_info_df['AC-WEIGHT'],
                       ac_info_df['track_id'],
                       ac_info_df['MODEL']]

    ac_info_df = ac_info_df.rename(columns={'lat': 'Latitude [deg]',
                                            'lon': 'Longitude [deg]',
                                            'geoaltitude': 'Altitude [m]',
                                            'velocity': 'Velocity [m/s]',
                                            # 'heading':'Heading [deg]',
                                            'turnrate': 'Turn Rate [deg/s]',
                                            'acceleration': 'Acceleration [m/s2]',
                                            'vertrate': 'Vertical Rate [m/s]'})


    fig1 = px.scatter_mapbox(ac_info_df, lat="Latitude [deg]", lon="Longitude [deg]", color="cluster", title=title_str,
                             hover_data=hover_data_list)
    fig1.update_traces(marker_size=1, mode="lines", opacity=0.9)

    results_path = r'C:/Users/nicol/Google Drive/PhD/Conferences & Papers/AIAA Aviation 2023/Code and results/2_2. Aircraft behaviour category detection/results visuals/'
    fig1.write_html(results_path + "2D_map_all_tracks.html")

# reference: https://towardsdatascience.com/interpretable-k-means-clusters-feature-importances-7e516eeb8d3c#7e14
def interpret_results_WCSS_Minimizers(features, centroids, path, ax=None, plt_kwargs={}):
    print('\n\ninterpret_results_WCSS_Minimizers')

    print('features = ', features)

    # print('cluster_centers shape = ', centroids.shape)
    # print('cluster_centers = ', centroids)

    # print('test1 =\n', centroids[0])
    # print('test2 =\n', centroids[1])
    # print('test3 =\n', centroids[2])
    # print('len(features) = ', len(features))
    # print('len(centroids) = ', len(centroids))

    # temp_centroids = [0 for x in range(len(centroids))]
    temp_centroids = [[0] * len(features) for i in range(len(centroids))]
    # print('initialization of temp_centroids = ', temp_centroids)

    for cluster_idx in range(0, len(centroids)):
        # print('cluster_idx = ', cluster_idx)
        # print('centroids[cluster_idx] =\n', centroids[cluster_idx])

        for feature_idx, feature in enumerate(centroids[cluster_idx]):
            # print('feature_idx =', feature_idx)
            # print('sum =', sum(feature))

            temp_centroids[cluster_idx][feature_idx] = round(sum(feature),2)

            # print('temp_centroids[cluster_idx] = ', temp_centroids[cluster_idx])
            # print('temp_centroids[cluster_idx][feature_idx] = ', temp_centroids[cluster_idx][feature_idx])

    # print('modified centroids =\n', temp_centroids)
    temp_centroids = np.array(temp_centroids)
    # print('modified centroids array =\n', temp_centroids)
    centroids = temp_centroids

    # print('centroids pre absolute =\n', centroids)
    centroids = abs(centroids)
    # print('centroids post absolute =\n', centroids)

    sorted_centroid_features_idx = centroids.argsort(axis=1)[:, ::-1]
    # print(f"\nSorted Feature/Dimension Indexes for Each Centroid in Descending Order: \n{sorted_centroid_features_idx}")

    sorted_centroid_features_values = np.take_along_axis(centroids, sorted_centroid_features_idx, axis=1)
    # print(f"Sorted Feature/Dimension Values for Each Centroid in Descending Order: \n{sorted_centroid_features_values}")
    # print('centroids =\n', centroids)

    # if features == ['lon', 'lat', 'alt']:
    #     df = pd.DataFrame(columns=['lon', 'lat', 'alt'])
    #     # print('trigger 1')
    # if features == ['lon', 'lat', 'alt', 'hdg']:
    #     df = pd.DataFrame(columns=['lon', 'lat', 'alt', 'hdg'])
    #     print('trigger 2')
    # df = pd.DataFrame(columns=['lon', 'lat', 'alt', 'hdg'])
    temp_list = []
    for cluster_idx in range(0, len(centroids)):
        first_features_in_centroid = centroids[cluster_idx][sorted_centroid_features_idx[cluster_idx]]
        # print('cluster_idx = ', cluster_idx)
        ordered_feat_imp = list(zip([features[feature] for feature in sorted_centroid_features_idx[cluster_idx]], first_features_in_centroid))
        # print(ordered_feat_imp)
        # df.loc[cluster_idx] = ordered_feat_imp
        # print('sorted_centroid_features_idx[cluster_idx] = ', sorted_centroid_features_idx[cluster_idx])
        # print('first_features_in_centroid = ', first_features_in_centroid)
        # print('first_features_in_centroid[xx] = ', first_features_in_centroid[0])
        # print('first_features_in_centroid[xx] = ', first_features_in_centroid[1])
        # if features == ['lon', 'lat', 'alt']:
        #     # df.loc[cluster_idx].lon = first_features_in_centroid[0]
        #     # df.loc[cluster_idx].lat = first_features_in_centroid[1]
        #     # df.loc[cluster_idx].alt = first_features_in_centroid[2]
        #     df.at[cluster_idx, 'lon'] = first_features_in_centroid[0]
        #     df.at[cluster_idx, 'lat'] = first_features_in_centroid[1]
        #     df.at[cluster_idx, 'alt'] = first_features_in_centroid[2]
        # if features == ['lon', 'lat', 'alt', 'hdg']:
        #     # df.loc[cluster_idx].lon = first_features_in_centroid[0]
        #     # df.loc[cluster_idx].lat = first_features_in_centroid[1]
        #     # df.loc[cluster_idx].alt = first_features_in_centroid[2]
        #     # df.loc[cluster_idx].hdg = first_features_in_centroid[3]
        #     df.at[cluster_idx, 'lon'] = first_features_in_centroid[0]
        #     df.at[cluster_idx, 'lat'] = first_features_in_centroid[1]
        #     df.at[cluster_idx, 'alt'] = first_features_in_centroid[2]
        #     df.at[cluster_idx, 'hdg'] = first_features_in_centroid[3]
        # df = df.append(ordered_feat_imp, ignore_index=False)
    #     temp_list.append(ordered_feat_imp)
    # print('temp_list =\n', temp_list)

    if features == ['lon', 'lat', 'alt']:
        # df = pd.DataFrame(temp_list, columns=['lon', 'lat', 'alt'])
        feature_imp_df = pd.DataFrame(centroids, columns=['lon', 'lat', 'alt'])
        row_labels = ['Longitude', 'Latitude', 'Altitude']
        feature_imp_df['Longitude Total'] = feature_imp_df['lon'].sum()
        feature_imp_df['Latitude Total'] = feature_imp_df['lat'].sum()
        feature_imp_df['Altitude Total'] = feature_imp_df['alt'].sum()
        # print('trigger 1')
    if features == ['lon', 'lat', 'alt', 'hdg']:
        # df = pd.DataFrame(temp_list, columns=['lon', 'lat', 'alt', 'hdg'])
        feature_imp_df = pd.DataFrame(centroids, columns=['lon', 'lat', 'alt', 'hdg'])
        row_labels = ['Longitude', 'Latitude', 'Altitude', 'Heading']
        feature_imp_df['Longitude Total'] = feature_imp_df['lon'].sum()
        feature_imp_df['Latitude Total'] = feature_imp_df['lat'].sum()
        feature_imp_df['Altitude Total'] = feature_imp_df['alt'].sum()
        feature_imp_df['Heading Total'] = feature_imp_df['hdg'].sum()
    print('feature importance results df =\n', feature_imp_df)
    feature_imp_df.to_csv(path +"nb clusters "+ str(len(centroids)) +" feature importance results.csv")

    if ax is None:
        # fig = plt.gcf().set_size_inches(10, 5)
        fig = plt.gcf()
        ax = plt.gca()

    # cols = features_str_list
    # rows = ['%d year' % x for x in (100, 50, 20, 10, 5)]
    cols = ['Cluster %d' % x for x in range(0, len(centroids))]
    # cols = ['Cluster %d' % x for x in range(0, len(features_str_list))]
    rows = features_str_list

    # colors = plt.cm.BuPu(np.linspace(0, 0.5, len(rows)))
    # colors = plt.cm.Set2(np.linspace(0, 0.5, len(rows)))
    colors = plt.cm.tab20c(np.linspace(0, 0.5, len(rows)))
    # n_rows = len(sorted_centroid_features_values)
    # n_rows = len(centroids)
    n_rows = len(rows)
    # n_rows = len(sorted_centroid_features_idx)
    # print('colors =', colors)
    # print('n_rows =', n_rows)

    # index = np.arange(len(cols)) + 0.3
    # index = np.arange(len(rows)) + 0.3
    index = np.arange(len(centroids)) + 0.3
    bar_width = 0.05

    y_offset = np.zeros(len(cols))
    # y_offset = np.zeros(len(rows))

    transpose_centroids = np.transpose(centroids)
    # print('transpose_centroids =\n', transpose_centroids)

    cell_text = []
    for row in range(n_rows):
        # print('row = ', row)
        # print('transpose_centroids =\n', transpose_centroids)
        # print('transpose_centroids[row] = ', transpose_centroids[row])
        # cell_text.append(['%1.1f' % (x / 1000.0) for x in y_offset])
        # cell_text.append(['%1.1f' % x for x in y_offset])
        # cell_text.append(['%1.1f' % x for x in sorted_centroid_features_values[row]])
        # cell_text.append(['%1.1f' % x for x in centroids[row]])
        cell_text.append(['%1.2f' % x for x in transpose_centroids[row]])
        # cell_text.append(['%1.1f' % x for x in list(df[row])])
        # print('cell_text =\n', cell_text)
        # cell_text.append([str(y_offset)])
    # Reverse colors and text labels to display the last value at the top.
    colors = colors[::-1]
    # cell_text.reverse()
    # the_table = ax.table(cellText=cell_text,
    #                       rowLabels=rows,
    #                       colLabels=cols,
    #                       loc='bottom')
    the_table = ax.table(cellText=cell_text,
                          rowLabels=row_labels,
                          rowColours=colors,
                          colLabels=cols,
                          loc='bottom')
    the_table.set_fontsize(20)
    the_table.scale(1, 2)

    for row in range(0, len(centroids)):
    # for row in range(n_rows):
    #     print('\n--------')
    #     print('cluster = ', row)
        # print('--------')
        # print('row  = ', row)
        # print('index = ', index)
        # print('index.shape = ', index.shape)
        # print('sorted_centroid_features_values[row] = ', sorted_centroid_features_values[row])
        # print('sorted_centroid_features_values[row].shape = ', sorted_centroid_features_values[row].shape)
        # print('y_offset = ', y_offset)
        # print('y_offset.shape = ', y_offset.shape)
        # print('centroids = ', centroids)
        # print('centroids[row] = ', centroids[row])
        # print('colors = ', colors)
        # print('colors[row] = ', colors[row])

        if row != n_rows:
            origin_index = index[row]
            # end_index = index[row+1]
            # temp_index = np.linspace(origin_index, end_index, len(centroids))
        else:
            origin_index = 3.3
            temp_index = index
        # print('origin_index = ', origin_index)
        # print('end_index = ', end_index)
        # print('temp_index = ', temp_index)

        for i, feat in enumerate(features):
            # print('====')
            # print('feat = ', feat)
            # print('temp_index = ', temp_index)
            # print('centroids[row][i] = ', centroids[row][i])
            # print('====')

            # ax.bar(index, sorted_centroid_features_values[row], bar_width, bottom=y_offset, color=colors[row])
            # ax.bar(index, centroids[row], bar_width, bottom=y_offset, color=colors[row])
            # ax.bar(temp_index, centroids[row][i], bar_width, color=colors[row], alpha=0.2, edgecolor='black')
            ax.bar(origin_index, centroids[row][i], bar_width, color=colors[i], edgecolor='black')
            # y_offset = y_offset + sorted_centroid_features_values[row]
            origin_index = origin_index + 0.1

        # label_y_offset = np.zeros(len(cols))
        # # print('label_y_offset = \n', label_y_offset)


        # for i, feat in enumerate(features):
        #     # print('+++')
        #     # print('i           = ', i)
        #     # print('feat        = ', feat)
        #     # print('row         = ', row)
        #     # print('index       = ', index)
        #     # print('y_offset    = ', y_offset)
        #     # print('x index     = ', index[row])
        #     # print('y index     = ', y_offset[i])
        #     # print('label_y_offset    = ', label_y_offset)
        #     # print('label_y_offset[i]    = ', label_y_offset[i])
        #     # ax.annotate(str(feat), xy=(index[row], y_offset[i]), fontsize=15)
        #     ax.annotate(str(feat), xy=(index[row], label_y_offset[i]), fontsize=15)
        #     # ax.annotate(str(feat), xy=(1, 0), xycoords='axes fraction', fontsize=15, horizontalalignment='right', verticalalignment='bottom')
        #     label_y_offset = label_y_offset + centroids[row]

        # ax.bar(index, sorted_centroid_features_idx[row], bar_width, bottom=y_offset, color=colors[row])
        # y_offset = y_offset + sorted_centroid_features_idx[row]



    plt.subplots_adjust(left=0.2, bottom=0.2)

    # ax.ylabel(f"Loss in ${value_increment}'s")
    # ax.yticks(values * value_increment, ['%d' % val for val in values])
    # ax.xticks([])
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

    fig.suptitle('Feature Importance')
    # ax.plot(range(0, len(alt_data_for_icao_idx)), alt_centroids, c='red', label='centroid', **plt_kwargs)
    # ax.bar(features_str_list, sorted_centroid_features_values[0], color='maroon', width=0.4)
    # # ax.legend(loc='upper right')
    # # ax.annotate(str(len(cluster_icao_idx)), xy=(1, 0), xycoords='axes fraction', fontsize=15, horizontalalignment='right', verticalalignment='bottom')
    # fig.suptitle('Cluster '+str(cluster_idx))
    # ax.set_xlabel('Time')
    # ax.set_ylabel('Altitude [m]')
    fig_name = path +"nb clusters "+ str(len(centroids)) +" feature importance plot.png"
    fig.savefig(fig_name, dpi=200)
    fig.clear(ax)
    plt.close(fig)

import os
# results_path = path + 'results/1 day results/'+str(n_clusters)+'clusters/'
# os.mkdir(results_path)
results_path = r'C:/Users/nicol/Google Drive/PhD/Conferences & Papers/AIAA Journal/Code and results/results/'
# used to delete all files in results diretory every time code is run
for f in os.listdir(results_path):
    try:
        os.remove(os.path.join(results_path, f))
    except PermissionError:
        continue

n_clusters_list, s_score_list, ssd_list, model_fit_time_list, s_score_calc_time_list, sil_vals_list = [], [], [], [], [], []
# for cluster_id in range(ncluster_range_min, ncluster_range_max):
# for cluster_id in np.arange(5, 50, 5):
cluster_track_id_df = pd.DataFrame()
s_core_results_df = pd.DataFrame()
n_cluster_loop_start_time = time.time()
# n_clusters_range = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50]
n_clusters_range = [30]
n_clusters_range = [5]
n_clusters_range = [5, 6, 7]
# n_clusters_range = [2]
n_clusters_range = [4]
# n_clusters_range = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]
# n_clusters_range = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
# n_clusters_range = [2, 3, 4]  # for debugging
# n_clusters_range = [2, 3, 4, 30]
iter = 1
for cluster_id in n_clusters_range:
# for cluster_id in n_clusters_range:
# for cluster_id in np.arange(5, 15, 5):
    print('----------------------------------------------------------------------')
    print(' Main loop for nb clusters', cluster_id, '(', str(iter), '/', str(len(n_clusters_range)), ')')
    print('----------------------------------------------------------------------')

    print('--> training nb clusters', str(cluster_id), 'kmeans model')
    model = multi_cluster_model(n_clusters=cluster_id, data=data_array)
    n_clusters_list.append(model[0])
    s_score_list.append(model[1])
    ssd_list.append(model[2])
    model_fit_time_list.append(model[3])
    s_score_calc_time_list.append(model[4])
    cluster_labels = model[5]
    cluster_centers = model[6]
    model_preds = model[7]
    sil_vals = model[8]
    # sil_vals_list.append([list(model[8])])

    temp_sil_vals_list = []
    for i, cluster in enumerate(np.unique(model_preds)):
        # sil_vals = list(model[8])
        sil_vals = model[8]
        cluster_silhouette_vals = sil_vals[model_preds == cluster]
        # temp_sil_vals_list.append(list(cluster_silhouette_vals))

        temp_sil_vals_list.append(round((sum(cluster_silhouette_vals)/len(cluster_silhouette_vals)), 4))

        # print('i =', i)
        # print('cluster =', cluster)
        # print('cluster_silhouette_vals =', cluster_silhouette_vals)
        # print('temp_sil_vals_list =', temp_sil_vals_list)
    sil_vals_list.append(temp_sil_vals_list)
    # print('sil_vals_list =', sil_vals_list)
    # sil_vals_df['sil_vals'] = temp_sil_vals_list
    # sil_vals_df.to_csv(results_path + 'cluster_'+str(cluster_id)+'_sil_vals.csv')

    # print('unique cluster labels =\n', np.unique(model[5], return_counts=True))
    unique_cluster_counts_df = pd.DataFrame(np.unique(cluster_labels, return_counts=True))
    unique_cluster_counts_df = unique_cluster_counts_df.drop(0)
    print('unique_cluster_counts_df =\n', unique_cluster_counts_df)
    # plt.bar(unique_cluster_counts_df.columns,unique_cluster_counts_df.values)

    # print('\n--> plotting nb clusters', str(cluster_id), 'distribution plot')
    # fig1 = plt.figure()
    # ax1 = fig1.add_subplot(111)
    # ax1.bar(unique_cluster_counts_df.columns, unique_cluster_counts_df.loc[1])
    # fig_name1 = results_path + "nb clusters "+str(cluster_id)+" distribution" + ".png"
    # fig1.savefig(fig_name1)
    # fig1.clear(ax1)
    # plt.close(fig1)

    print('--> plotting nb clusters', str(cluster_id), 'computing cost plot')
    model_fit_time_list
    fig777 = plt.figure()
    ax777 = fig777.add_subplot(111)
    ax777.set_title('Computing Cost vs Number of Clusters')
    # print('iter = ', iter)
    print('n_clusters_range = ', n_clusters_range)
    # print('n_clusters_range[0:iter] = ', n_clusters_range[0:iter])
    temp_n_clusters_range = n_clusters_range[:iter]
    # print('temp_n_clusters_range = ', temp_n_clusters_range)
    print('model_fit_time_list = ', model_fit_time_list)
    # ax777.plot(n_clusters_range, model_fit_time_list, linewidth=3)
    ax777.plot(temp_n_clusters_range, model_fit_time_list, linewidth=3)
    ax777.set_xlabel('Number of Clusters K')
    ax777.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax777.set_ylabel('Computing Time [s]')
    fig_name777 = results_path + "all nb clusters computing cost plot" + ".png"
    fig777.savefig(fig_name777)
    fig777.clear(ax777)
    plt.close(fig777)

    # color_list = cm.Set2(np.linspace(0, 1, cluster_id))
    color_list = cm.Set3(np.linspace(0, 1, cluster_id))
    # color_list = cm.rainbow(np.linspace(0, 1, cluster_id))
    # color_list = cm.gist_rainbow(np.linspace(0, 1, cluster_id))
    # print('np.linspace(0, 1, cluster_idx) = ', np.linspace(0, 1, cluster_id))
    # print('color_list =\n', color_list)
    # print('color_list len =', len(color_list))

    print('--> plotting nb clusters', str(cluster_id), 'silhouette plot')
    silhouette_plot(multitimeseries_data=data_array, model_preds=model_preds, cluster_idx=cluster_id, centroids=cluster_centers, color_list=color_list, sil_vals=sil_vals, path=results_path)
    # # silhouette_plot(multitimeseries_data=data_array, model_preds=model[7], cluster_idx=cluster_id, centroids=model[6], sil_vals=model[1], path=results_path)

    # # for 1 plot per cluster
    # print('--> plotting nb clusters', str(cluster_id), 'multivariate timeseries plot')
    # multivariate_timeseries_plot(multitimeseries_data=data_array, model_preds=model_preds, cluster_idx=cluster_id, centroids=cluster_centers, path=results_path)

    # # for 1 plot for all clusters (lat long, alt only)
    # print('--> plotting nb clusters', str(cluster_id), 'multivariate timeseries plot2')
    # multivariate_timeseries_plot2(multitimeseries_data=data_array, model_preds=model_preds, features=features_str_list,cluster_idx=cluster_id, centroids=cluster_centers, color_list=color_list, path=results_path)

    # # for 1 plot for all clusters (lat long, alt, vrate, accel, turnrate)
    # print('--> plotting nb clusters', str(cluster_id), 'multivariate timeseries plot2')
    # multivariate_timeseries_plot3(multitimeseries_data=data_array, model_preds=model_preds, cluster_idx=cluster_id, centroids=cluster_centers, path=results_path)

    print('--> plotting nb clusters', str(cluster_id), 'multivariate timeseries plot4')
    multivariate_timeseries_plot4(multitimeseries_data=data_array, model_preds=model_preds, features=features_str_list,cluster_idx=cluster_id, centroids=cluster_centers, color_list=color_list, path=results_path)

    # print('--> plotting nb clusters', str(cluster_id), 'twod_traj_plot')
    # twod_traj_plot(ac_info_df=ac_info_df,multitimeseries_data=data_array, model_preds=model_preds, cluster_idx=cluster_id, centroids=cluster_centers, path=results_path)

    track_id_array = np.array(track_id_list)
    for cluster_idx in range(0, cluster_id):
        # print('cluster_id = ', cluster_id)
        # print('cluster_idx = ', cluster_idx)
        # cluster_icao_idx = np.where(model[5] == cluster_id)[0]

        if cluster_idx == 0:
            cluster_track_id_df = pd.DataFrame() # used to empty previous results from dataframe
            cluster_track_id_df['cluster_'+str(cluster_idx)] = track_id_array[np.where(cluster_labels == cluster_idx)[0]]
        else:
            temp_df = pd.DataFrame(track_id_array[np.where(cluster_labels == cluster_idx)[0]], columns=['cluster_'+str(cluster_idx)])
            cluster_track_id_df = pd.concat([cluster_track_id_df, temp_df], axis=1)
        # print('cluster_track_id_df =\n', cluster_track_id_df)

        # print('cluster_track_id_df =\n', cluster_track_id_df)
        cluster_track_id_df.to_csv(results_path + 'nb clusters ' + str(cluster_id) + '_results.csv')

    # this bit is used to save the centroids in individual csv files for post analysis using 'plot_results_visuals' code
    cluster_centers = scaler.inverse_transform(cluster_centers.reshape(cluster_centers.shape[0], -1)).reshape(cluster_centers.shape)

    if len(cluster_centers) == 4:
        cluster_lon_centroid_df = pd.DataFrame({'cluster_0':cluster_centers[0][0],
                                                'cluster_1':cluster_centers[1][0],
                                                'cluster_2':cluster_centers[2][0],
                                                'cluster_3':cluster_centers[3][0]})
        cluster_lat_centroid_df = pd.DataFrame({'cluster_0': cluster_centers[0][1],
                                                'cluster_1': cluster_centers[1][1],
                                                'cluster_2': cluster_centers[2][1],
                                                'cluster_3': cluster_centers[3][1]})
        cluster_alt_centroid_df = pd.DataFrame({'cluster_0': cluster_centers[0][2],
                                                'cluster_1': cluster_centers[1][2],
                                                'cluster_2': cluster_centers[2][2],
                                                'cluster_3': cluster_centers[3][2]})
    if len(cluster_centers) == 5:
        cluster_lon_centroid_df = pd.DataFrame({'cluster_0': cluster_centers[0][0],
                                                'cluster_1': cluster_centers[1][0],
                                                'cluster_2': cluster_centers[2][0],
                                                'cluster_3': cluster_centers[3][0],
                                                'cluster_4': cluster_centers[4][0]})
        cluster_lat_centroid_df = pd.DataFrame({'cluster_0': cluster_centers[0][1],
                                                'cluster_1': cluster_centers[1][1],
                                                'cluster_2': cluster_centers[2][1],
                                                'cluster_3': cluster_centers[3][1],
                                                'cluster_4': cluster_centers[4][1]})
        cluster_alt_centroid_df = pd.DataFrame({'cluster_0': cluster_centers[0][2],
                                                'cluster_1': cluster_centers[1][2],
                                                'cluster_2': cluster_centers[2][2],
                                                'cluster_3': cluster_centers[3][2],
                                                'cluster_4': cluster_centers[4][2]})
    if len(cluster_centers) == 6:
        cluster_lon_centroid_df = pd.DataFrame({'cluster_0': cluster_centers[0][0],
                                                'cluster_1': cluster_centers[1][0],
                                                'cluster_2': cluster_centers[2][0],
                                                'cluster_3': cluster_centers[3][0],
                                                'cluster_4': cluster_centers[4][0],
                                                'cluster_5': cluster_centers[5][0]})
        cluster_lat_centroid_df = pd.DataFrame({'cluster_0': cluster_centers[0][1],
                                                'cluster_1': cluster_centers[1][1],
                                                'cluster_2': cluster_centers[2][1],
                                                'cluster_3': cluster_centers[3][1],
                                                'cluster_4': cluster_centers[4][1],
                                                'cluster_5': cluster_centers[5][1]})
        cluster_alt_centroid_df = pd.DataFrame({'cluster_0': cluster_centers[0][2],
                                                'cluster_1': cluster_centers[1][2],
                                                'cluster_2': cluster_centers[2][2],
                                                'cluster_3': cluster_centers[3][2],
                                                'cluster_4': cluster_centers[4][2],
                                                'cluster_5': cluster_centers[5][2]})
    if len(cluster_centers) == 7:
        cluster_lon_centroid_df = pd.DataFrame({'cluster_0': cluster_centers[0][0],
                                                'cluster_1': cluster_centers[1][0],
                                                'cluster_2': cluster_centers[2][0],
                                                'cluster_3': cluster_centers[3][0],
                                                'cluster_4': cluster_centers[4][0],
                                                'cluster_5': cluster_centers[5][0],
                                                'cluster_6': cluster_centers[6][0],
                                                })
        cluster_lat_centroid_df = pd.DataFrame({'cluster_0': cluster_centers[0][1],
                                                'cluster_1': cluster_centers[1][1],
                                                'cluster_2': cluster_centers[2][1],
                                                'cluster_3': cluster_centers[3][1],
                                                'cluster_4': cluster_centers[4][1],
                                                'cluster_5': cluster_centers[5][1],
                                                'cluster_6': cluster_centers[6][1],
                                                })
        cluster_alt_centroid_df = pd.DataFrame({'cluster_0': cluster_centers[0][2],
                                                'cluster_1': cluster_centers[1][2],
                                                'cluster_2': cluster_centers[2][2],
                                                'cluster_3': cluster_centers[3][2],
                                                'cluster_4': cluster_centers[4][2],
                                                'cluster_5': cluster_centers[5][2],
                                                'cluster_6': cluster_centers[6][2],
                                                })

    # print('cluster_lon_centroid_df = \n', cluster_lon_centroid_df)
    # print('cluster_lat_centroid_df = \n', cluster_lat_centroid_df)
    # print('cluster_alt_centroid_df = \n', cluster_alt_centroid_df)
    cluster_lon_centroid_df.to_csv(results_path+'nb clusters '+str(cluster_id)+'_lon_centroids.csv')
    cluster_lat_centroid_df.to_csv(results_path+'nb clusters '+str(cluster_id)+'_lat_centroids.csv')
    cluster_alt_centroid_df.to_csv(results_path+'nb clusters '+str(cluster_id)+'_alt_centroids.csv')

    ac_info_data = ac_info_n_cluster_data_merger(ac_info_data=ac_info_data, cluster_track_id_df=cluster_track_id_df)

    # # for 1 plot for all clusters (lat long, alt only)
    # print('--> plotting nb clusters', str(cluster_id), 'multivariate timeseries plot2')
    # multivariate_timeseries_plot3(multitimeseries_data=data_array, model_preds=model_preds, features=features_str_list,
    #                               ac_info_data=ac_info_data, cluster_idx=cluster_id, centroids=cluster_centers, color_list=color_list,
    #                               path=results_path)

    # for 1 plot of heading values even if not used in training model
    print('--> plotting nb clusters', str(cluster_id), 'heading_timeseries_plot')
    heading_timeseries_plot(features=features_str_list, multitimeseries_data=data_array, ac_info_data=ac_info_data, colors=color_list, model_preds=model_preds, cluster_idx=cluster_id, centroids=cluster_centers, path=results_path, ax=None)
    #
    # for 1 plot of altitude values even if not used in training model
    print('--> plotting nb clusters', str(cluster_id), 'altitude_timeseries_plot')
    altitude_timeseries_plot(features=features_str_list, multitimeseries_data=data_array, ac_info_data=ac_info_data,colors=color_list, model_preds=model_preds, cluster_idx=cluster_id, centroids=cluster_centers,path=results_path, ax=None)

    s_core_results_df = pd.DataFrame()
    s_core_results_df['n_clusters'] = n_clusters_list
    s_core_results_df['s_score'] = s_score_list
    s_core_results_df['ssd'] = ssd_list
    s_core_results_df['model_fit_time [s]'] = model_fit_time_list
    s_core_results_df['s_score_calc_time time [s]'] = s_score_calc_time_list
    # print('sil_vals_list =', sil_vals_list)
    s_core_results_df['silvals'] = sil_vals_list

    features_df = pd.DataFrame({'features':features_str_list})
    s_core_results_df = pd.concat([s_core_results_df, features_df], axis=1)
    print('s_core_results_df =\n', s_core_results_df)
    s_core_results_df.to_csv(results_path+'s_score_results.csv')

    n_cluster_loop_end_time = (time.time() - n_cluster_loop_start_time)
    print('n_cluster_loop_end_time    = ' + str(display_time(n_cluster_loop_end_time, 5)))
    iter += 1

# features_df = pd.DataFrame({'features':features_str_list})
# s_core_results_df = pd.concat([s_core_results_df,features_df],axis=1)
print('s_core_results_df =\n', s_core_results_df)
s_core_results_df.to_csv(results_path + 's_score_results.csv')

# s_core_results_df = pd.DataFrame()
# s_core_results_df['n_clusters'] = n_clusters_list
# s_core_results_df['s_score'] = s_score_list
# s_core_results_df['ssd'] = ssd_list
# s_core_results_df['model_fit_time [s]'] = model_fit_time_list
# s_core_results_df['s_score_calc_time time [s]'] = s_score_calc_time_list
# # s_core_results_df['features'] = features_str_list
# # s_core_results_df.at[1, 'features'] = features_str_list
# features_df = pd.DataFrame({'features':features_str_list})
# s_core_results_df = pd.concat([s_core_results_df,features_df],axis=1)


print('s_core_results_df =\n', s_core_results_df)
s_core_results_df.to_csv(results_path+'s_score_results.csv')


# fig1_999 = plt.figure()
# ax1_999 = fig1_999.add_subplot(111)
# ax1_999.plot(n_clusters_range, silhouette_avg, linewidth=3)
# ax1_999.set_xlabel('Values of K')
# ax1_999.xaxis.set_major_locator(MaxNLocator(integer=True))
# ax1_999.set_ylabel('Silhouette score')
# ax1_999.set_title('Silhouette analysis For Optimal k')
# fig_name1_999 = results_path + "all nb clusters silhouette analysis plot.png"
# fig1_999.savefig(fig_name1_999)
# fig1_999.clear(ax1_999)
# plt.close(fig1_999)
#
# fig1_888 = plt.figure()
# ax1_888 = fig1_888.add_subplot(111)
# ax1_888.plot(n_clusters_range, sum_of_squared_distances, linewidth=3)
# ax1_888.set_xlabel('Values of K')
# ax1_888.xaxis.set_major_locator(MaxNLocator(integer=True))
# ax1_888.set_ylabel('Sum of squared distances/Inertia')
# ax1_888.set_title('Elbow Method For Optimal k')
# fig_name1_888 = results_path + "all nb clusters elbow method plot.png"
# fig1_888.savefig(fig_name1_888)
# fig1_888.clear(ax1_888)
# plt.close(fig1_888)
#
# print('--> plotting nb clusters', str(cluster_id), 'computing cost plot')
# model_fit_time_list
# fig777 = plt.figure()
# ax777 = fig777.add_subplot(111)
# ax777.set_title('Computing Cost vs Number of Clusters')
# ax777.plot(n_clusters_range, model_fit_time_list, linewidth=3)
# ax777.set_xlabel('Number of Clusters K')
# ax777.xaxis.set_major_locator(MaxNLocator(integer=True))
# ax777.set_ylabel('Computing Time [s]')
# fig_name777 = results_path + "all nb clusters computing cost plot" + ".png"
# fig777.savefig(fig_name777)
# fig777.clear(ax777)
# plt.close(fig777)

# plt.show()

# ac_info_df = pd.read_csv(path+'flight tracks/final data 1 day/smooth_tracks_dataset_all.csv')
# # ac_info_df = ac_info_df.loc[:, ~ac_info_df.columns.str.contains('^Unnamed')]
# print('ac_info_df = \n', ac_info_df)

# unique_track_ids_list = []
# for col in cluster_track_id_df.columns:
#     temp = list(cluster_track_id_df[col].unique())
#     # print('temp = \n', temp)
#     unique_track_ids_list = unique_track_ids_list + temp
# # print('unique_track_ids_list = \n', unique_track_ids_list)
#
# ac_info_df['cluster'] = 0
# # ac_info_df.loc[ac_info_df.track_id == 'AA9A0A_1', 'cluster'] = "cluster_0"
# for unique_track in unique_track_ids_list:
#     for col in cluster_track_id_df.columns:
#         if (unique_track in cluster_track_id_df[col].unique()) == True:
#             # print('---------------')
#             # print(unique_track+' is in '+str(col))
#             ac_info_df.loc[ac_info_df.track_id == unique_track, 'cluster'] = str(col)
#
# ac_info_df = ac_info_df[ac_info_df.cluster != 0]
#
# # ac_info_df['cluster'] = np.where(adsb_df['track'] == data_df)
# print('ac_info_df = \n', ac_info_df)


def plot_2D_centroids_all_clusters(lon_df, lat_df, ac_info_df, path, ax=None, plt_kwargs={}):

    # n_clusters = len(pd.unique(df['cluster']))
    n_clusters = len(lon_df.columns)
    # print('n_clusters = ', n_clusters)

    color_list = cm.Set3(np.linspace(0, 1, n_clusters))
    # print('color_list = ', color_list)
    # print('len color_list = ', len(color_list))

    # fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig, (ax1) = plt.subplots(1, 1)
    opacity = 0.7
    linewidth = 4
    zorder = 10
    edgecolor = 'k'
    for i in range(0, n_clusters):
        # print('i = ', i)
        temp_ac_info_df = ac_info_df[ac_info_df['cluster'] == ('cluster_'+str(i))]
        # print('temp_ac_info_df = \n', temp_ac_info_df)
        # ax1.scatter(x=lon_df['cluster_'+str(i)], y=lat_df['cluster_'+str(i)], s=3, alpha=1, color=color_list[i])
        # ax1.scatter(x=lon_df['cluster_'+str(i)], y=lat_df['cluster_'+str(i)], s=3, alpha=1, color=color_list[i], zorder=10, path_effects=[pe.Stroke(linewidth=linewidth+2, foreground=edgecolor), pe.Normal()])
        # ax1.scatter(x=temp_ac_info_df['lon'], y=temp_ac_info_df['lat'], s=1, alpha=0.8, color=color_list[i], zorder=1)
        ax1.plot(lon_df['cluster_'+str(i)], lat_df['cluster_'+str(i)], linewidth=linewidth, alpha=1, color=color_list[i], zorder=10, path_effects=[pe.Stroke(linewidth=linewidth+2, foreground=edgecolor), pe.Normal()], label=str('Centroid of Cluster '+str(i)))
        # ax1.plot(temp_ac_info_df['lon'], temp_ac_info_df['lat'], linewidth=1, alpha=0.8, color=color_list[i], zorder=1)
        for j, track_id in enumerate(np.unique(temp_ac_info_df['track_id'])):
            # print('j = ', j)
            # print('track_id = ', track_id)
            temp_trackid_df = temp_ac_info_df[temp_ac_info_df['track_id'] == track_id]
            # print('temp_trackid_df = \n', temp_trackid_df)
            ax1.plot(temp_trackid_df['lon'], temp_trackid_df['lat'], linewidth=1, alpha=0.8, color=color_list[i], zorder=1)

    airport_marker_size = 70
    airport_marker_zorder = 50
    # airport_marker_color = 10
    airport_name_fontsize = 20
    airport_name_offset = 0.0001
    # JFK
    apt_lon = -73.778900
    apt_lat = 40.639801
    # ax1.scatter(apt_lon, apt_lat, s=airport_marker_size, alpha=1, zorder=airport_marker_zorder, marker="*", label=str('JFK'))
    ax1.scatter(apt_lon, apt_lat, s=airport_marker_size, alpha=1, zorder=airport_marker_zorder, marker="*", color='red', label=str('Airport'))
    ax1.annotate(str('JFK'), xy=(apt_lon-airport_name_offset*apt_lon, apt_lat+airport_name_offset*apt_lat), xycoords='data', zorder=airport_marker_zorder, fontsize=airport_name_fontsize, horizontalalignment='left', verticalalignment='bottom')
    # EWR
    apt_lon = -74.174462
    apt_lat = 40.689531
    ax1.scatter(apt_lon, apt_lat, s=airport_marker_size, alpha=1, zorder=airport_marker_zorder, marker="*", color='red')
    ax1.annotate(str('EWR'), xy=(apt_lon-airport_name_offset*apt_lon, apt_lat+airport_name_offset*apt_lat), xycoords='data', zorder=airport_marker_zorder, fontsize=airport_name_fontsize, horizontalalignment='right', verticalalignment='bottom')
    # TEB
    apt_lon = -74.0615292
    apt_lat = 40.858332
    ax1.scatter(apt_lon, apt_lat, s=airport_marker_size, alpha=1, zorder=airport_marker_zorder, marker="*", color='red')
    ax1.annotate(str('TEB'), xy=(apt_lon-airport_name_offset*apt_lon, apt_lat+airport_name_offset*apt_lat), xycoords='data', zorder=airport_marker_zorder, fontsize=airport_name_fontsize, horizontalalignment='left', verticalalignment='bottom')
    # LGA
    apt_lon = -73.872597
    apt_lat = 40.777199
    ax1.scatter(apt_lon, apt_lat, s=airport_marker_size, alpha=1, zorder=airport_marker_zorder, marker="*", color='red')
    ax1.annotate(str('LGA'), xy=(apt_lon-airport_name_offset*apt_lon, apt_lat+airport_name_offset*apt_lat), xycoords='data', zorder=airport_marker_zorder, fontsize=airport_name_fontsize, horizontalalignment='left', verticalalignment='bottom')
    # PHL
    apt_lon = -75.241096
    apt_lat = 39.871899
    ax1.scatter(apt_lon, apt_lat, s=airport_marker_size, alpha=1, zorder=airport_marker_zorder, marker="*", color='red')
    ax1.annotate(str('PHL'), xy=(apt_lon-airport_name_offset*apt_lon, apt_lat+airport_name_offset*apt_lat), xycoords='data', zorder=airport_marker_zorder, fontsize=airport_name_fontsize, horizontalalignment='left', verticalalignment='bottom')
    # BWI
    apt_lon = -76.668297
    apt_lat = 39.1753998
    ax1.scatter(apt_lon, apt_lat, s=airport_marker_size, alpha=1, zorder=airport_marker_zorder, marker="*", color='red')
    ax1.annotate(str('BWI'), xy=(apt_lon-airport_name_offset*apt_lon, apt_lat+airport_name_offset*apt_lat), xycoords='data', zorder=airport_marker_zorder, fontsize=airport_name_fontsize, horizontalalignment='left', verticalalignment='bottom')
    # ACY
    apt_lon = -74.577202
    apt_lat = 39.4575996
    ax1.scatter(apt_lon, apt_lat, s=airport_marker_size, alpha=1, zorder=airport_marker_zorder, marker="*", color='red')
    ax1.annotate(str('ACY'), xy=(apt_lon-airport_name_offset*apt_lon, apt_lat+airport_name_offset*apt_lat), xycoords='data', zorder=airport_marker_zorder, fontsize=airport_name_fontsize, horizontalalignment='left', verticalalignment='bottom')

    # ABE
    apt_lon = -75.440804
    apt_lat = 40.6520996
    ax1.scatter(apt_lon, apt_lat, s=airport_marker_size, alpha=1, zorder=airport_marker_zorder, marker="*", color='red')
    ax1.annotate(str('ABE'), xy=(apt_lon - airport_name_offset * apt_lon, apt_lat + airport_name_offset * apt_lat),xycoords='data', zorder=airport_marker_zorder, fontsize=airport_name_fontsize, horizontalalignment='left',verticalalignment='bottom')

    # MDT
    apt_lon = -76.763397
    apt_lat = 40.1935005
    ax1.scatter(apt_lon, apt_lat, s=airport_marker_size, alpha=1, zorder=airport_marker_zorder, marker="*", color='red')
    ax1.annotate(str('MDT'), xy=(apt_lon-airport_name_offset*apt_lon, apt_lat+airport_name_offset*apt_lat), xycoords='data', zorder=airport_marker_zorder, fontsize=airport_name_fontsize, horizontalalignment='left', verticalalignment='bottom')




    ax1.legend(loc='upper right').set_zorder(100)

    ax1.title.set_text('Centroids 2D Map')
    ax1.set_xlabel('Longitude [deg]')
    ax1.set_ylabel('Latitude [deg]')

    fig_name = path + "nb clusters "+str(n_clusters)+"_2D_centroids_map.png"
    # plt.show()
    fig.savefig(fig_name, dpi=200)

n_clusters = 4
n_clusters = n_clusters_range[0]

for cluster_id in n_clusters_range:
    print('--> plotting nb clusters', str(cluster_id), 'plot_2D_centroids_all_clusters')

    centroids_lon_df = pd.read_csv(results_path+'nb clusters '+str(cluster_id)+'_lon_centroids.csv')
    centroids_lat_df = pd.read_csv(results_path+'nb clusters '+str(cluster_id)+'_lat_centroids.csv')
    centroids_alt_df = pd.read_csv(results_path+'nb clusters '+str(cluster_id)+'_alt_centroids.csv')
    centroids_lon_df = centroids_lon_df.loc[:, ~centroids_lon_df.columns.str.contains('^Unnamed')]
    centroids_lat_df = centroids_lat_df.loc[:, ~centroids_lat_df.columns.str.contains('^Unnamed')]
    centroids_alt_df = centroids_alt_df.loc[:, ~centroids_alt_df.columns.str.contains('^Unnamed')]
    # print('centroids_lon_df =\n', centroids_lon_df)
    # print('centroids_lat_df =\n', centroids_lat_df)
    # print('centroids_alt_df =\n', centroids_alt_df)

    # print('--> plotting nb clusters', str(n_clusters), 'plot_2D_centroids_all_clusters')
    plot_2D_centroids_all_clusters(lon_df=centroids_lon_df, lat_df=centroids_lat_df, ac_info_df=ac_info_data, path=results_path)

import winsound
frequency = 700  # Set Frequency To 2500 Hertz
duration = 500  # Set Duration To 1000 ms == 1 second
winsound.Beep(frequency, duration)
frequency = 1500
duration = 500
winsound.Beep(frequency, duration)
frequency = 700
duration = 500
winsound.Beep(frequency, duration)

plt.show()
