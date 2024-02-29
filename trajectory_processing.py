
# https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
px.set_mapbox_access_token('pk.eyJ1IjoibmljdmIiLCJhIjoiY2thNzBxMnl0MDAyYzJ0bmZpeW1jOHNlayJ9.p5h0jJ78qIUWcRLQ19muYw')
import sktime
from time import sleep
from tqdm import tqdm
import datetime
from scipy.signal import savgol_filter

# set pandas dataframe display properties
max_rows = 100
max_cols = 15
# max_cols = None
pd.set_option('display.width', 320)
pd.set_option('display.max_columns', max_cols)
pd.set_option('display.min_rows', max_rows)
pd.set_option('display.max_rows', max_rows)

# data_selector = '30 min'
# data_selector = '1 h'
# data_selector = '12 h'
# data_selector = '24 h'
data_selector = '2 days'
# data_selector = '1 week'

# output_dir = r'C:/Users/nicol/Google Drive/PhD/Coding/Python scripts/1. Determine what is the right data to collect/histograms/'
output_dir = r'C:/Users/nicol/Google Drive/PhD/Conferences & Papers/AIAA Aviation 2023/Code and results/2. Collect and process the selected data/'
paper_dir = r'C:/Users/nicol/Google Drive/PhD/Conferences & Papers/AIAA Aviation 2023/Code and results/2_2. Aircraft behaviour category detection/'

if data_selector == '30 min':
    adsb_df = pd.read_csv(output_dir+'histogram data_2018-01-01 09-00-00_2018-01-01 09-29-41.csv')  # 30 min
if data_selector == '1 h':
    adsb_df = pd.read_csv(output_dir+'histogram data_2018-01-01 09-00-00_2018-01-01 09-59-21.csv')  # 1 h
if data_selector == '12 h':
    adsb_df = pd.read_csv(output_dir+'histogram data_2018-01-01 09-00-00_2018-01-01 21-00-01.csv')  # 12 h
if data_selector == '24 h':
    adsb_df = pd.read_csv(output_dir+'histogram data_2018-01-01 09-00-00_2018-01-02 08-45-41.csv')  # 24 h
if data_selector == '2 days':
    adsb_df = pd.read_csv(output_dir+'histogram data_2018-01-01 09-00-00_2018-01-03 08-31-31.csv')  # 2 days
if data_selector == '1 week':
    adsb_df = pd.read_csv(output_dir+'histogram data_xxxxxxxxxxxxxxxx.csv')  # 1 week

print("raw data =\n", adsb_df)

def remove_cols_containing_substring(df, substring):
    data_col = df.columns
    cols = list(data_col.values)
    # print("cols =", cols)
    cols_to_remove = []
    for col_name in cols:
        if substring not in col_name:
            continue
        else:
            cols_to_remove.append(str(col_name))
    # print("cols =", cols)
    # print("cols_to_remove =", cols_to_remove)
    # cols_to_remove = ['Unnamed: 0']
    cols = [x for x in cols if x in cols_to_remove]  # removes all cols except the ones we need
    data_col = data_col.drop(cols)
    df = df[data_col[:]]
    return df

# adsb_df = remove_cols_containing_substring(adsb_df, "Unnamed")

adsb_df = adsb_df.loc[:, ~adsb_df.columns.str.contains('^Unnamed')]
data_col = adsb_df.columns
cols = list(data_col.values)
# print("cols =", cols)
# cols_to_remove = ['Unnamed: 0']
# cols_to_remove = ['TYPE AIRCRAFT']
# cols_to_remove = ['time', 'icao24', 'local time']
# cols_to_remove = ['time', 'icao24', 'local time', 'TYPE AIRCRAFT']
# cols_to_remove = ['time', 'local time', 'lat', 'lon']
# cols_to_remove = ['time', 'icao24', 'local time', 'lat', 'lon']
# cols = [x for x in cols if x in cols_to_remove]  # removes all cols except the ones we need
# data_col = data_col.drop(cols)
adsb_df = adsb_df[data_col[:]].dropna()
# adsb_df['lat'].round(decimals=5)
# adsb_df['lon'].round(decimals=5)
# adsb_df['geoaltitude'].round(decimals=2)
# adsb_df['velocity'].round(decimals=2)
# adsb_df['heading'].round(decimals=2)
# adsb_df['vertrate'].round(decimals=2)

# print("adsb_df 2 cols = \n", list(adsb_df.columns))
# print("adsb_df 2 = \n", adsb_df)

# adsb_df = adsb_df.rename(columns={'lat': 'Latitude [deg]',
#                                             'lon': 'Longitude [deg]',
#                                             'geoaltitude': 'Altitude [m]',
#                                             'velocity': 'Velocity [m/s]',
#                                             'heading': 'Heading [deg]',
#                                             'vertrate': 'Vertical Rate [m/s]'})


# adsb_df = adsb_df[adsb_df['icao24'] == 'A5BB1F']
# adsb_df = adsb_df[adsb_df['icao24'] == 'A04E60']

# print("adsb_df 3 = \n", adsb_df)
# adsb_df = adsb_df.sample(100)
# adsb_df = adsb_df.sample(1000)
# adsb_df = adsb_df.sample(10000)
print('removed nans, col containing unnamed and cols_to_remove')
print("adsb_df 1 = \n", adsb_df)

path_of_2021_data = r"C:/Users/nicol/Google Drive/PhD/Coding/Python scripts/2. Collect and process the selected data/data collection and processing scripts/aircraft registration processing/processed_data_2021.csv"
ac_data_2021 = pd.read_csv(path_of_2021_data, skip_blank_lines=True)
ac_data_2021 = ac_data_2021.loc[:, ~ac_data_2021.columns.str.contains('^Unnamed')]
print("raw data_2021 =\n", ac_data_2021)
ac_data_2021.rename(columns={'MODE S CODE HEX': 'icao24'}, inplace=True)
# ac_data_2021['icao24'].str.upper()
data_col = ac_data_2021.columns
cols = list(data_col.values)
# print("cols =", cols)
# cols_to_remove = ['Unnamed: 0']
cols_to_remove = ['SERIAL NUMBER','TYPE-ACFT', 'TYPE-ENG', 'AC-CAT', 'BUILD-CERT-IND', 'NO-ENG', 'SPEED', 'MFR MDL CODE','TC-DATA-SHEET', 'TC-DATA-HOLDER', 'ENG MFR MDL', 'YEAR MFR', 'STATUS CODE', 'MODE S CODE', 'FRACT OWNER', 'AIR WORTH DATE', 'TYPE REGISTRANT', 'NAME', 'STREET', 'STREET2', 'CITY', 'STATE', 'ZIP CODE', 'REGION', 'COUNTY', 'OTHER NAMES(1)', 'OTHER NAMES(2)', 'OTHER NAMES(3)', 'OTHER NAMES(4)', 'OTHER NAMES(5)', 'EXPIRATION DATE', 'UNIQUE ID', 'KIT MFR', ' KIT MODEL', 'COUNTRY', 'LAST ACTION DATE', 'CERT ISSUE DATE', 'LAST ACTION DATE', 'MODE S CODE', 'UNIQUE ID', 'BUILD-CERT-IND']
cols_to_remove = ['CERTIFICATION', 'AC-WEIGHT', 'N-NUMBER', 'TYPE ENGINE', 'MFR', 'MODEL', 'NO-SEATS', 'SERIAL NUMBER','TYPE-ACFT', 'TYPE-ENG', 'AC-CAT', 'BUILD-CERT-IND', 'NO-ENG', 'SPEED', 'MFR MDL CODE','TC-DATA-SHEET', 'TC-DATA-HOLDER', 'ENG MFR MDL', 'YEAR MFR', 'STATUS CODE', 'MODE S CODE', 'FRACT OWNER', 'AIR WORTH DATE', 'TYPE REGISTRANT', 'NAME', 'STREET', 'STREET2', 'CITY', 'STATE', 'ZIP CODE', 'REGION', 'COUNTY', 'OTHER NAMES(1)', 'OTHER NAMES(2)', 'OTHER NAMES(3)', 'OTHER NAMES(4)', 'OTHER NAMES(5)', 'EXPIRATION DATE', 'UNIQUE ID', 'KIT MFR', ' KIT MODEL', 'COUNTRY', 'LAST ACTION DATE', 'CERT ISSUE DATE', 'LAST ACTION DATE', 'MODE S CODE', 'UNIQUE ID', 'BUILD-CERT-IND']
cols_to_remove = ['CERTIFICATION', 'TYPE AIRCRAFT','AC-WEIGHT', 'N-NUMBER', 'TYPE ENGINE', 'MFR', 'NO-SEATS', 'SERIAL NUMBER','TYPE-ACFT', 'TYPE-ENG', 'AC-CAT', 'BUILD-CERT-IND', 'NO-ENG', 'SPEED', 'MFR MDL CODE','TC-DATA-SHEET', 'TC-DATA-HOLDER', 'ENG MFR MDL', 'YEAR MFR', 'STATUS CODE', 'MODE S CODE', 'FRACT OWNER', 'AIR WORTH DATE', 'TYPE REGISTRANT', 'NAME', 'STREET', 'STREET2', 'CITY', 'STATE', 'ZIP CODE', 'REGION', 'COUNTY', 'OTHER NAMES(1)', 'OTHER NAMES(2)', 'OTHER NAMES(3)', 'OTHER NAMES(4)', 'OTHER NAMES(5)', 'EXPIRATION DATE', 'UNIQUE ID', 'KIT MFR', ' KIT MODEL', 'COUNTRY', 'LAST ACTION DATE', 'CERT ISSUE DATE', 'LAST ACTION DATE', 'MODE S CODE', 'UNIQUE ID', 'BUILD-CERT-IND']
cols_to_remove = ['TYPE AIRCRAFT', 'N-NUMBER', 'MFR', 'SERIAL NUMBER','TYPE-ACFT', 'TYPE-ENG', 'AC-CAT', 'BUILD-CERT-IND', 'SPEED', 'MFR MDL CODE','TC-DATA-SHEET', 'TC-DATA-HOLDER', 'ENG MFR MDL', 'YEAR MFR', 'STATUS CODE', 'MODE S CODE', 'FRACT OWNER', 'AIR WORTH DATE', 'TYPE REGISTRANT', 'NAME', 'STREET', 'STREET2', 'CITY', 'STATE', 'ZIP CODE', 'REGION', 'COUNTY', 'OTHER NAMES(1)', 'OTHER NAMES(2)', 'OTHER NAMES(3)', 'OTHER NAMES(4)', 'OTHER NAMES(5)', 'EXPIRATION DATE', 'UNIQUE ID', 'KIT MFR', ' KIT MODEL', 'COUNTRY', 'LAST ACTION DATE', 'CERT ISSUE DATE', 'LAST ACTION DATE', 'MODE S CODE', 'UNIQUE ID', 'BUILD-CERT-IND']
cols = [x for x in cols if x in cols_to_remove]  # removes all cols except the ones we need
data_col = data_col.drop(cols)
ac_data_2021 = ac_data_2021[data_col[:]].dropna()
print("ac_data_2021 =\n", ac_data_2021)

adsb_df['icao24'] = adsb_df['icao24'].astype("string")
adsb_df['icao24'] = adsb_df['icao24'].str.upper()
# adsb_df = pd.merge(adsb_df, ac_data_2021, on='icao24', how='left')
# adsb_df = pd.merge(adsb_df, ac_data_2021, on='icao24', how='outer')
adsb_df = pd.merge(adsb_df, ac_data_2021, on='icao24')
print('merged ac_data_2021 and adsb_df dataframes into adsb_df')
print("adsb_df 2 =\n", adsb_df)

# convert time to datetime objects
def unix_to_local(unix_time):
    local_time = datetime.datetime.fromtimestamp(unix_time)
    return local_time

def unix_to_utc(unix_time):
    # utc_time = datetime.datetime.utcfromtimestamp(unix_time).strftime('%Y-%m-%d %H:%M:%S')
    utc_time = datetime.datetime.utcfromtimestamp(unix_time)
    return utc_time

# create new col of datetime objects used for all time operations
# time_series_data['datetime'] = pd.to_datetime(time_series_data['time'])
adsb_df['datetime'] = adsb_df['time'].apply(unix_to_local)
# adsb_df = adsb_df[['datetime', 'icao24', 'geoaltitude']]
print('added datetime objects to adsb_df')
print('adsb_df 3 =\n', adsb_df)

# remove obvious outliers based on boxplot outliers
adsb_df = adsb_df[adsb_df['geoaltitude'] < 17000]
adsb_df = adsb_df[adsb_df['velocity'] < 375]
vrate_threshold = 20
adsb_df = adsb_df[(adsb_df['vertrate'] > -vrate_threshold) & (adsb_df['vertrate'] < vrate_threshold)]
# adsb_df = adsb_df[(adsb_df['vertrate'] < -vrate_threshold) | (adsb_df['vertrate'] > vrate_threshold)]
adsb_df = adsb_df.drop(adsb_df[(adsb_df['icao24'] == 'AA8FDA') & (adsb_df['local time'] == '2018-01-02 12:50:41')].index)
adsb_df = adsb_df.drop(adsb_df[(adsb_df['icao24'] == 'A343C4') & (adsb_df['local time'] == '2018-01-01 13:33:41')].index)
adsb_df = adsb_df.drop(adsb_df[(adsb_df['icao24'] == 'A4B916') & (adsb_df['local time'] == '2018-01-02 10:48:01')].index)
# df_new = df.drop(df[(df['col_1'] == 1.0) & (df['col_2'] == 0.0)].index)
print('adsb_df 4 =\n', adsb_df)

# def calc_turn_rate(time, hdg):
#     # utc_time = datetime.datetime.utcfromtimestamp(unix_time).strftime('%Y-%m-%d %H:%M:%S')
#     turnrate = datetime.datetime.utcfromtimestamp(time)
#     return turnrate

# adsb_df['turnrate'] = adsb_df['heading'].apply(unix_to_local)
# adsb_df['turnrate'] = adsb_df.apply(lambda x: calc_turn_rate(x.time, x.heading), axis=1)
adsb_df['diff hdg'] = adsb_df['heading'].diff()
adsb_df['diff time'] = adsb_df['time'].diff()
adsb_df['turnrate'] = adsb_df['diff hdg'] / adsb_df['diff time']
adsb_df = adsb_df.drop(['diff hdg', 'diff time'], axis=1)
print('adsb_df 5 =\n', adsb_df)


adsb_df['diff vel'] = adsb_df['velocity'].diff()
adsb_df['diff time'] = adsb_df['time'].diff()
adsb_df['acceleration'] = adsb_df['diff vel'] / adsb_df['diff time']
adsb_df = adsb_df.drop(['diff vel', 'diff time'], axis=1)
print('adsb_df 6 =\n', adsb_df)


# adsb_df = adsb_df.drop(['heading'], axis=1)
# print('adsb_df 7 =\n', adsb_df)

# ++++++++++++++++++++++++++++++++++++++++
# priint

# # for debugging
# # adsb_df = adsb_df[adsb_df['icao24'].isin(list(['A343C4', 'A4B973']))]  # 1 rotorcraft with repeated bad signals (middle signal is stuck since diff = 0)
# adsb_df = adsb_df[adsb_df['icao24'].isin(list(['AD35ED', 'A4B57E', 'ADE5D4', 'AC90BE', 'ADD352']))]

#
# adsb_df = adsb_df.set_axis(adsb_df['datetime'], axis=0)
# adsb_df = adsb_df.drop(['time', 'local time'], axis=1)
# print('change index to datetime objects and drop time and local time cols')
# print('adsb_df 4 =\n', adsb_df)
# print('adsb_df 4 info =\n', adsb_df.info())
#
# # adsb_df = adsb_df.col_name.resample('M').mean()
# # adsb_df = adsb_df.resample('1Min').mean()
# adsb_df = adsb_df.groupby('icao24').resample('1Min').mean()
# # adsb_df = adsb_df.groupby([pd.Grouper(freq='1Min'), 'icao24'])
# # adsb_df['Event'].count().unstack()
# # adsb_df = adsb_df.unstack('icao24', fill_value=999.99)
# adsb_df = adsb_df.unstack('icao24')
#
# print('rescale datetime objects for average of every 1min intervals')
# print('adsb_df 5 =\n', adsb_df)
# print('adsb_df 5 info =\n', adsb_df.info())

# adsb_df = adsb_df[adsb_df['icao24'].isin(list(['A04AA2', 'A5BB1F', 'A1F471', 'A099A7']))]
# adsb_df = adsb_df[adsb_df['icao24'].isin(list(['AB318C', 'A6A45E', 'A06067', 'A05281', 'A35365', 'A32321', 'A65560', 'A32F9D']))]
# adsb_df = adsb_df[adsb_df['icao24'].isin(list(['AB318C', 'A6A45E']))]
# adsb_df = adsb_df[adsb_df['icao24'] == 'A65560']  # has 2 clearly defined separate tracks
# adsb_df = adsb_df[adsb_df['icao24'] == 'AD35ED']  # has 1 bad cutoff_signal datapoints that needs removing
# adsb_df = adsb_df[adsb_df['icao24'] == 'A1F471']  # has 2 bad cutoff_signal datapoints that needs removing
# adsb_df = adsb_df[adsb_df['icao24'].isin(list(['AD35ED', 'A4B57E', 'ADE5D4', 'AC90BE', 'ADD352']))]
# adsb_df = adsb_df[adsb_df['icao24'].isin(list(['A343C4', 'A4B973']))]  # 1 rotorcraft with repeated bad signals (middle signal is stuck since diff = 0)


rotorcraft_df = adsb_df[adsb_df['TYPE AIRCRAFT'].isin(list(['Rotorcraft']))]
fws_df = adsb_df[adsb_df['TYPE AIRCRAFT'].isin(list(['Fixed wing single engine']))]
fwm_df = adsb_df[adsb_df['TYPE AIRCRAFT'].isin(list(['Fixed wing multi engine']))]
# print('rotorcraft_df =\n', rotorcraft_df)
# print('fws_df =\n', fws_df)
# print('fwm_df =\n', fwm_df)

print('len rotorcraft_df = ', len(rotorcraft_df))
print('len fws_df        = ', len(fws_df))
print('len fwm_df        = ', len(fwm_df))

nb_unique_rotorcraft = rotorcraft_df['icao24'].nunique()
nb_unique_fws = fws_df['icao24'].nunique()
nb_unique_fwm = fwm_df['icao24'].nunique()

icao_list = pd.unique(adsb_df['icao24'])
# print('unique icaos =\n', icao_list)

print('nb unique rotorcraft = ', nb_unique_rotorcraft)
print('nb unique fws        = ', nb_unique_fws)
print('nb unique fwm        = ', nb_unique_fwm)
print('nb of unique icaos   = ', len(icao_list))

# ac_id = 'A04E60'
# temp_data = adsb_df[adsb_df['icao24'] == ac_id]
# fig1, axs1 = plt.subplots(6)
# fig1.suptitle(ac_id)
# axs1[0].set_title('lat')
# axs1[0].plot(temp_data['local time'], temp_data['lat'])
# axs1[1].set_xlabel('lon')
# axs1[1].plot(temp_data['local time'], temp_data['lon'])
# axs1[2].set_title('geoaltitude')
# axs1[2].plot(temp_data['local time'], temp_data['geoaltitude'])
# axs1[3].set_title('velocity')
# axs1[3].plot(temp_data['local time'], temp_data['velocity'])
# axs1[4].set_title('heading')
# axs1[4].plot(temp_data['local time'], temp_data['heading'])
# axs1[5].set_title('vertrate')
# axs1[5].plot(temp_data['local time'], temp_data['vertrate'])
#
# ac_id = 'A5BB1F'
# temp_data = adsb_df[adsb_df['icao24'] == ac_id]
# fig2, axs2 = plt.subplots(6)
# fig1.suptitle(ac_id)
# axs2[0].set_title('lat')
# axs2[0].plot(temp_data['local time'], temp_data['lat'])
# axs2[1].set_xlabel('lon')
# axs2[1].plot(temp_data['local time'], temp_data['lon'])
# axs2[2].set_title('geoaltitude')
# axs2[2].plot(temp_data['local time'], temp_data['geoaltitude'])
# axs2[3].set_title('velocity')
# axs2[3].plot(temp_data['local time'], temp_data['velocity'])
# axs2[4].set_title('heading')
# axs2[4].plot(temp_data['local time'], temp_data['heading'])
# axs2[5].set_title('vertrate')
# axs2[5].plot(temp_data['local time'], temp_data['vertrate'])

# ac_id1 = 'A04AA2'
# ac_id2 = 'A5BB1F'
# ac_id3 = 'A1F471'
# ac_id4 = 'A099A7'
# temp_data1 = adsb_df[adsb_df['icao24'] == ac_id1]
# temp_data2 = adsb_df[adsb_df['icao24'] == ac_id2]
# temp_data3 = adsb_df[adsb_df['icao24'] == ac_id3]
# temp_data4 = adsb_df[adsb_df['icao24'] == ac_id4]
# ac_model1 = list(temp_data1['MODEL'])[0]
# ac_model2 = list(temp_data2['MODEL'])[0]
# ac_model3 = list(temp_data3['MODEL'])[0]
# ac_model4 = list(temp_data4['MODEL'])[0]
# print("ac_model1 =\n", ac_model1)
# fig1, axs1 = plt.subplots(6)
# fig1.suptitle(ac_id1)
# axs1[0].set_title('lat')
# axs1[0].plot(temp_data1['local time'], temp_data1['lat'], label=ac_model1)
# axs1[0].plot(temp_data2['local time'], temp_data2['lat'], label=ac_model2)
# axs1[0].plot(temp_data3['local time'], temp_data3['lat'], label=ac_model3)
# axs1[0].plot(temp_data4['local time'], temp_data4['lat'], label=ac_model4)
# axs1[0].legend(loc='upper left')
# axs1[1].set_xlabel('lon')
# axs1[1].plot(temp_data1['local time'], temp_data1['lon'], label=ac_model1)
# axs1[1].plot(temp_data2['local time'], temp_data2['lon'], label=ac_model2)
# axs1[1].plot(temp_data3['local time'], temp_data3['lon'], label=ac_model3)
# axs1[1].plot(temp_data4['local time'], temp_data4['lon'], label=ac_model4)
# axs1[1].legend(loc='upper left')
# axs1[2].set_title('geoaltitude')
# axs1[2].plot(temp_data1['local time'], temp_data1['geoaltitude'], label=ac_model1)
# axs1[2].plot(temp_data2['local time'], temp_data2['geoaltitude'], label=ac_model2)
# axs1[2].plot(temp_data3['local time'], temp_data3['geoaltitude'], label=ac_model3)
# axs1[2].plot(temp_data4['local time'], temp_data4['geoaltitude'], label=ac_model4)
# axs1[2].legend(loc='upper left')
# axs1[3].set_title('velocity')
# axs1[3].plot(temp_data1['local time'], temp_data1['velocity'], label=ac_model1)
# axs1[3].plot(temp_data2['local time'], temp_data2['velocity'], label=ac_model2)
# axs1[3].plot(temp_data3['local time'], temp_data3['velocity'], label=ac_model3)
# axs1[3].plot(temp_data4['local time'], temp_data4['velocity'], label=ac_model4)
# axs1[3].legend(loc='upper left')
# axs1[4].set_title('heading')
# axs1[4].plot(temp_data1['local time'], temp_data1['heading'], label=ac_model1)
# axs1[4].plot(temp_data2['local time'], temp_data2['heading'], label=ac_model2)
# axs1[4].plot(temp_data3['local time'], temp_data3['heading'], label=ac_model3)
# axs1[4].plot(temp_data4['local time'], temp_data4['heading'], label=ac_model4)
# axs1[4].legend(loc='upper left')
# axs1[5].set_title('vertrate')
# axs1[5].plot(temp_data1['local time'], temp_data1['vertrate'], label=ac_model1)
# axs1[5].plot(temp_data2['local time'], temp_data2['vertrate'], label=ac_model2)
# axs1[5].plot(temp_data3['local time'], temp_data3['vertrate'], label=ac_model3)
# axs1[5].plot(temp_data4['local time'], temp_data4['vertrate'], label=ac_model4)
# axs1[5].legend(loc='upper left')
#
# for ax in axs1.flat:
#     ax.set(xlabel='x-label')
#
# # Hide x labels and tick labels for top plots and y ticks for right plots.
# for ax in axs1.flat:
#     ax.label_outer()

def PlotAircraftTimeSeries(data, ac_id):

    # temp_data = data[data['icao24'] == 'A04E60']
    temp_data = data[data['icao24'] == ac_id]
    # print('temp data =\n', temp_data)

    fig6 = plt.figure()
    fig7 = plt.figure()
    fig8 = plt.figure()
    fig9 = plt.figure()
    fig10 = plt.figure()
    fig11 = plt.figure()
    ax6 = fig6.add_subplot(111)
    ax7 = fig7.add_subplot(111)
    ax8 = fig8.add_subplot(111)
    ax9 = fig9.add_subplot(111)
    ax10 = fig10.add_subplot(111)
    ax11 = fig11.add_subplot(111)

    ax6.plot(temp_data['local time'], temp_data['lat'], label='lat')
    ax7.plot(temp_data['local time'], temp_data['lon'], label='lon')
    ax8.plot(temp_data['local time'], temp_data['geoaltitude'], label='geoaltitude')
    ax9.plot(temp_data['local time'], temp_data['velocity'], label='velocity')
    ax10.plot(temp_data['local time'], temp_data['heading'], label='heading')
    ax11.plot(temp_data['local time'], temp_data['vertrate'], label='vertrate')

    plt.legend(loc='best')

    fig_name6 = paper_dir+"/flight tracks/"+"lat_"+str(ac_id)+".png"
    fig6.savefig(fig_name6)
    fig6.clear()
    plt.close(fig6)

    fig_name7 = paper_dir+"/flight tracks/"+"lon_"+str(ac_id)+".png"
    fig7.savefig(fig_name7)
    fig7.clear()
    plt.close(fig7)

    fig_name8 = paper_dir+"/flight tracks/"+"alt_"+str(ac_id)+".png"
    fig8.savefig(fig_name8)
    fig8.clear()
    plt.close(fig8)

    fig_name9 = paper_dir+"/flight tracks/"+"velocity_"+str(ac_id)+".png"
    fig9.savefig(fig_name9)
    fig9.clear()
    plt.close(fig9)

    fig_name10 = paper_dir+"/flight tracks/"+"heading_"+str(ac_id)+".png"
    fig10.savefig(fig_name10)
    fig10.clear()
    plt.close(fig10)

    fig_name11 = paper_dir+"/flight tracks/"+"vertrate_"+str(ac_id)+".png"
    fig11.savefig(fig_name11)
    fig11.clear()
    plt.close(fig11)

def PlotAircraftTimeSeriesCaseStudy(data, ac_id, path):

    # temp_data = data[data['icao24'] == 'A04E60']
    temp_data = data[data['icao24'] == ac_id]
    # print('temp data =\n', temp_data)

    fig6 = plt.figure()
    fig7 = plt.figure()
    fig8 = plt.figure()
    fig9 = plt.figure()
    fig10 = plt.figure()
    fig11 = plt.figure()
    ax6 = fig6.add_subplot(111)
    ax7 = fig7.add_subplot(111)
    ax8 = fig8.add_subplot(111)
    ax9 = fig9.add_subplot(111)
    ax10 = fig10.add_subplot(111)
    ax11 = fig11.add_subplot(111)

    ax6.plot(temp_data['local time'], temp_data['lat'], label='lat')
    ax7.plot(temp_data['local time'], temp_data['lon'], label='lon')
    ax8.plot(temp_data['local time'], temp_data['geoaltitude'], label='geoaltitude')
    ax9.plot(temp_data['local time'], temp_data['velocity'], label='velocity')
    ax10.plot(temp_data['local time'], temp_data['heading'], label='heading')
    ax11.plot(temp_data['local time'], temp_data['vertrate'], label='vertrate')

    plt.legend(loc='best')

    fig_name6 = path+"lat_"+str(ac_id)+".png"
    fig6.savefig(fig_name6)
    fig6.clear()
    plt.close(fig6)

    fig_name7 = path+"lon_"+str(ac_id)+".png"
    fig7.savefig(fig_name7)
    fig7.clear()
    plt.close(fig7)

    fig_name8 = path+"alt_"+str(ac_id)+".png"
    fig8.savefig(fig_name8)
    fig8.clear()
    plt.close(fig8)

    fig_name9 = path+"velocity_"+str(ac_id)+".png"
    fig9.savefig(fig_name9)
    fig9.clear()
    plt.close(fig9)

    fig_name10 = path+"heading_"+str(ac_id)+".png"
    fig10.savefig(fig_name10)
    fig10.clear()
    plt.close(fig10)

    fig_name11 = path+"vertrate_"+str(ac_id)+".png"
    fig11.savefig(fig_name11)
    fig11.clear()
    plt.close(fig11)

case_study_path = paper_dir+"/flight tracks/same cruise route/"
case_study_path = paper_dir+"/flight tracks/same approach/"
case_study_path = paper_dir+"/flight tracks/all processed tracks/"
case_study_path = paper_dir+"/flight tracks/debug/"
# case_study_path = paper_dir+"/flight tracks/debug2/"

# for ele in tqdm(icao_list):
#     # PlotAircraftTimeSeries(data=adsb_df, ac_id=ele)
#     # PlotAircraftTimeSeries(data=adsb_df, ac_id=ele)
#     PlotAircraftTimeSeriesCaseStudy(data=adsb_df, ac_id=ele, path=case_study_path)

# PlotAircraftTimeSeries2(data=adsb_df, ac_id='A04E60')
# PlotAircraftTimeSeries(data=adsb_df, ac_id='A04E60')
# PlotAircraftTimeSeries(data=adsb_df, ac_id='A5BB1F')
# plt.show()


# # print('adsb_df =\n', adsb_df)
# adsb_df['track'] = 1
# print('adsb_df =\n', adsb_df)
#
# # adsb_df.loc[adsb_df.geoaltitude < 100, 'track'] = "2"
# # print('adsb_df =\n', adsb_df)
# print('iloc')
# print(adsb_df.loc[adsb_df.geoaltitude < 100])
# print(adsb_df.loc[adsb_df.geoaltitude < 100].index)
# # index of last row that have < 100 alt
# print(adsb_df.loc[adsb_df.geoaltitude < 100].index[-1])
# # adsb_df['track'][92438:] = 2
# # adsb_df['track'] = adsb_df['track'].mask(adsb_df['track'] > 15, 0)
# # adsb_df['track'] = np.where(adsb_df['track'].between(8,11), 0, adsb_df['track'])
# adsb_df.at[92438, 'track'] = 999
# # adsb_df[-92438:]['track'] = 2
# print(len(adsb_df))
# # adsb_df['track'] = adsb_df.iloc[92438:92602]['track'] = 2
# adsb_df.loc[92439:92602,"track"]=2
# print('adsb_df =\n', adsb_df)

def SplitTracks(data_df, cutoff_time_s, cutoff_alt_m, cutoff_min_datapoints, cutoff_pct):
    # finds all unique icao codes in data_df
    unique_icaos = list(pd.unique(data_df['icao24']))
    unique_icaos = [x for x in unique_icaos if str(x) != 'nan']
    # print('unique_icaos =\n', unique_icaos)

    # drop poor signal rows in data_df where datapoint jumps too far across 2 successive rows to be realistic
    data_df['test'] = 1
    for icao in unique_icaos:
        temp_df = data_df[data_df['icao24'] == icao]
        # print('before temp_df '+str(icao)+'=\n', temp_df)
        # print(data_df['geoaltitude'].diff())
        # # print(data_df['geoaltitude'].diff() > cutoff_alt_m)
        #
        # cutoff_indexes_list = list((data_df['geoaltitude'].diff() > cutoff_signal).index)
        # print('cutoff_indexes_list =\n', cutoff_indexes_list)
        #
        # print('cutoff_indexes_df =\n', data_df.loc[cutoff_indexes_list])
        # # data_df.loc[data_df['geoaltitude'].diff() > cutoff_alt_m, 'track'] = 888

        # data_df.loc[data_df['geoaltitude'].diff() > cutoff_signal, 'test'] = 999
        # data_df.loc[(data_df['geoaltitude'].diff() > cutoff_signal) & (data_df['geoaltitude'].diff(periods=-1) > cutoff_signal), 'test'] = 999
        # print('pct_change() '+str(icao)+' =\n', temp_df['geoaltitude'].pct_change())
        # print('pct_change(periods=-1) '+str(icao)+' =\n', temp_df['geoaltitude'].pct_change(periods=-1))
        temp_df.loc[abs(temp_df['geoaltitude'].pct_change()) > cutoff_pct, 'test'] = 888
        # temp_df.loc[(abs(temp_df['geoaltitude'].pct_change()) == 0) & (temp_df['geoaltitude'].isin()), 'test'] = 888
        temp_df.loc[abs(temp_df['geoaltitude'].pct_change(periods=-1)) > cutoff_pct, 'test'] = 999
        # temp_df.loc[(abs(temp_df['geoaltitude'].pct_change()) > cutoff_pct) & (abs(temp_df['geoaltitude'].pct_change(periods=-1)) > cutoff_pct), 'test'] = 777
        # print('--> removed datapoints because of cutoff_pct larger than '+str(cutoff_pct)+'=\n', temp_df[temp_df['test'] == 888], temp_df[temp_df['test'] == 999])
        # print('after data_df =\n', data_df.loc[[41612, 41613, 41614, 41615, 41616]])
        # plot_idx = 4043
        # if icao == 'A4B973':
        #     plot_idx = 58137
        #     print('test after data_df A4B973 =\n', temp_df.loc[[plot_idx - 2, plot_idx - 1, plot_idx, plot_idx + 1, plot_idx + 2]])
        # if icao == 'ADE5D4':
        #     plot_idx = 19661
        #     print('test after data_df ADE5D4 =\n', temp_df.loc[[plot_idx - 2, plot_idx - 1, plot_idx, plot_idx + 1, plot_idx + 2]])
        # try:
        #     plot_idx = temp_df[temp_df['test'] == 999].index[0]
        #     print('test after data_df 999 =\n', temp_df.loc[[plot_idx-2, plot_idx-1, plot_idx, plot_idx+1, plot_idx+2]])
        # except IndexError:
        #     continue
        # print('after temp_df =\n', temp_df)
        data_df.update(temp_df)

    data_df = data_df[data_df.test != 999]
    data_df = data_df[data_df.test != 888]
    data_df = data_df.drop('test', axis=1)

    # initially give a value of 1 to all tracks for all aircraft
    data_df['track'] = 1
    data_df['track_id'] = 1

    temp_df = 0

    # iterates over each unique icao aircraft and splits into a new track if:
    # 1. 1 same track skips time interval > than xxxx
    # 2.
    for icao in unique_icaos:
        track_id = 1
        temp_df = data_df[data_df['icao24'] == icao]
        # print('temp_df for icao = ', str(icao),'\n', temp_df)
        # print('altitudes below 100 for ', str(icao),'=\n')
        # ground_alt_list = list(temp_df.loc[temp_df.geoaltitude < 100].index)
        # print(ground_alt_list)
        #
        # # make a new track if elevation below 100
        # # starts with all previous icao track = 1 and changes all values after ground_alt_list[0] to track = 2
        # if ground_alt_list != []:
        #     print('--> aircraft is grounded!')
        #     track_id += 1
        #     print('track_id = ', track_id)
        #     print('ground_alt_list[0]  = ', ground_alt_list[0])
        #     print('ground_alt_list[-1] = ', ground_alt_list[-1])
        #     print('temp_df[-1] = ', list(temp_df.tail(1).index)[0], 41748)
        #     temp_df.loc[ground_alt_list[0]:list(temp_df.tail(1).index)[0], "track"] = track_id
        # else:
        #     print('--> no times in flight that the aircraft is grounded')
        #     continue
        # # make a new track if elevation below 100 and time lapse between rows is more than 1 index
        # # xxxxxxx description xxxxx
        # split_id = 1
        # prev_split_id = 1
        # prev_idx = ground_alt_list[0]
        # for idx in ground_alt_list[1:]:
        #     print('-------')
        #     print('idx      =', idx)
        #     print('prev_idx =', prev_idx)
        #     diff = idx-prev_idx
        #     print('diff =', diff)
        #     if diff > 1:
        #         split_id += 1
        #         track_id += 1
        #         temp_df.loc[prev_idx+1:idx+1, "track"] = track_id
        #
        #     print('split_id =', split_id)
        #     prev_idx = idx
        #     prev_split_id = split_id
        # print(temp_df['time'].diff())
        # print(temp_df['time'].diff() > cutoff_time_s)
        # print(list((temp_df['time'].diff() > cutoff_time_s).index))
        # print('test gizmo =\n', temp_df['geoaltitude'].pct_change())
        # find all temp_df indexes where time skips more than 10 over successive rows
        # give temporary value of 999 for now to identify them
        temp_df.loc[temp_df['time'].diff() > cutoff_time_s, 'track'] = 999
        # temp_df.loc[temp_df['time'].diff() > cutoff_time_s, 'track'] = 999
        # temp_df.loc[temp_df['geoaltitude'].diff() > cutoff_alt_m, 'track'] = 888
        # temp_df.loc[(temp_df['time'].diff() > cutoff_time_s) & ~(temp_df['geoaltitude'].diff() > cutoff_alt_m), 'track'] = 999
        temp_df.loc[(temp_df['time'].diff() > cutoff_time_s) & (temp_df['geoaltitude'].diff() > cutoff_alt_m), 'track'] = 999
        # print('--> removed datapoints because of cutoff_time_s & cutoff_alt_m =\n', temp_df[temp_df['track'] == 999])

        # create split_ranges as a list of the diff index ranges used to separate each track
        # each range in split_ranges is separated based on each 999 value
        # initialize split_ranges with only the start and end indexes, then append to it all the detected 999 ranges
        if not temp_df.empty:
            split_ranges = [list(temp_df.head(1).index)[0], list(temp_df.tail(1).index)[0]]
        # test_split_ranges = [list(temp_df.head(1).index)[0], list(temp_df.tail(1).index)[0]]
        for index, row in temp_df.iterrows():
            if (row['track'] == 999) or (row['track'] == 888):
                # print('-------')
                # print('idx   =', index)
                # print('track =', row['track'])
                # print('icao  =', row['icao24'])
                # print('time  =', row['local time'])
                split_ranges.append(index)
                # test_split_ranges.append(index)
                # test_split_ranges.append(index-1)
                # test_split_ranges.append(index-2)
                # test_split_ranges.append(index+1)
                # test_split_ranges.append(index+2)
        # print('split_ranges =\n', split_ranges)
        sorted_split_ranges = sorted(split_ranges)
        # print('sorted_split_ranges =\n', sorted_split_ranges)
        # print('sorted_split_ranges_df =\n', temp_df.loc[sorted_split_ranges])

        # test_sorted_split_ranges = sorted(test_split_ranges)
        # print('test_sorted_split_ranges =\n', test_sorted_split_ranges)
        # print('test_sorted_split_ranges_df =\n', temp_df.loc[test_sorted_split_ranges])



        # will go over all ranges in sorted_split_ranges and add a new track for each in col track
        range_idx = 1
        max_range_idx = len(sorted_split_ranges)-1
        # print('max_range_idx =', max_range_idx)
        for ele in range(0, max_range_idx):
            # print('ele =', ele)
            # for any condition other than track 1
            if ele != 0:
                track_id += 1
                # print('sorted_split_ranges[ele] =', sorted_split_ranges[ele])
                # print('sorted_split_ranges[ele+1] =', sorted_split_ranges[ele+1])
                temp_df.loc[sorted_split_ranges[ele]:sorted_split_ranges[ele+1], "track"] = round(track_id, 0)


        # print(list(temp_df['time'].diff() > 10))
        # temp_df.loc[ground_alt_list[0]:list(temp_df.tail(1).index)[0], "track"] = track_id
        # print('temp_df for icao = ', str(icao),'\n', temp_df)

        # add a new track_id col that combines icao24 and track into 1 col
        # required to distinguish ids between diff tracks from diff ac manifesting diff behaviour patterns
        temp_df['track'] = temp_df['track'].astype(int)
        temp_df['track_id'] = temp_df['icao24'].astype(str)+'_'+temp_df['track'].astype(str)
        # print('dtypes =\n', temp_df.dtypes)
        # print('temp_df for icao = ', str(icao),'\n', temp_df)
        data_df.update(temp_df)

        # # ----------------------------------------------------------------------------------------------
        # # add part here to deal with situations where you have a track that has only 1 to 3 time datapoint
        # # just need to detect it and delete these tracks
        # # print('unique track_id =\n', temp_df["track_id"].value_counts())
        # # print('unique track_id =\n', temp_df["track_id"].nunique())
        # print('unique track_id =\n', temp_df.track_id.value_counts())
        # # print('unique track_id =\n', temp_df.track_id.value_counts()[0])
        # # print('unique track_id =\n', temp_df.track_id.value_counts()[1])
        # # print('unique track_id =\n', list(temp_df.track_id.value_counts().index))
        # track_idx = 0
        # for track in list(temp_df.track_id.value_counts().index):
        #     # print('++++++++')
        #     # print('track = ', track)
        #     if temp_df.track_id.value_counts()[track_idx] <= cutoff_min_datapoints:
        #         print('--> removed '+str(track)+' cols because track has less than '+str(cutoff_min_datapoints)+' datapoints')
        #         # temp_df = temp_df[temp_df.track_id != track]
        #         # temp_df = temp_df.drop('track_id', axis=1)
        #         # temp_df = temp_df.loc[:, ~temp_df.columns.str.contains(str(track))]
        #         data_df = data_df[data_df["track_id"].str.match(str(track)) == False]
        #     track_idx += 1

        # ----------------------------------------------------------------------------------------------



        # data_df = data_df[data_df.track_id != 1]

        # data_df['track'] = data_df['track'].map(temp_df.set_index('track')['track'])


        # track_id += 1

    # ----------------------------------------------------------------------------------------------
    # add part here to deal with situations where you have a track that has only 1 to 3 time datapoint
    # just need to detect it and delete these tracks
    # print('unique track_id =\n', temp_df["track_id"].value_counts())
    # print('unique track_id =\n', temp_df["track_id"].nunique())
    print('unique track_id =\n', data_df.track_id.value_counts())
    # print('unique track_id =\n', temp_df.track_id.value_counts()[0])
    # print('unique track_id =\n', temp_df.track_id.value_counts()[1])
    # print('unique track_id =\n', list(temp_df.track_id.value_counts().index))
    track_idx = 0
    for track in list(data_df.track_id.value_counts().index):
        # print('++++++++')
        # print('track = ', track)
        try:
            if data_df.track_id.value_counts()[track_idx] <= cutoff_min_datapoints:
                # print('--> removed ' + str(track) + ' cols because track has less than ' + str(
                #     cutoff_min_datapoints) + ' datapoints')
                # temp_df = temp_df[temp_df.track_id != track]
                # temp_df = temp_df.drop('track_id', axis=1)
                # temp_df = temp_df.loc[:, ~temp_df.columns.str.contains(str(track))]
                data_df = data_df[data_df["track_id"].str.match(str(track)) == False]
        except IndexError:
            continue
        track_idx += 1
    # print('data_df =\n', data_df)
    # print('data_df dtypes =\n', data_df.dtypes)
    data_df['track'] = data_df['track'].astype(int)
    # print('data_df dtypes =\n', data_df.dtypes)
    # print('data_df =\n', data_df)
    return data_df

adsb_df = SplitTracks(data_df=adsb_df, cutoff_time_s=3*60, cutoff_alt_m=200, cutoff_min_datapoints=1, cutoff_pct=0.1)
adsb_df.to_csv(case_study_path+'smooth_tracks_dataset.csv')
# adsb_df.to_csv(case_study_path+'smooth_tracks_dataset_test.csv')

fig1 = px.scatter_mapbox(adsb_df, lat="lat", lon="lon", color="icao24", hover_data=[adsb_df['icao24'], adsb_df['local time'], adsb_df['geoaltitude'], adsb_df['velocity'], adsb_df['heading'], adsb_df['vertrate'], adsb_df['TYPE AIRCRAFT'], adsb_df['MODEL']])
fig1.update_traces(marker_size = 2, mode="lines")

# fig2 = px.scatter_3d(adsb_df, x="lon", y="lat", z="geoaltitude", color="icao24", hover_data=[adsb_df['icao24'], adsb_df['local time'], adsb_df['geoaltitude'], adsb_df['velocity'], adsb_df['heading'], adsb_df['vertrate'], adsb_df['TYPE AIRCRAFT'], adsb_df['MODEL']])
fig2 = px.scatter_3d(adsb_df, x="lon", y="lat", z="geoaltitude", color="track_id", hover_data=[adsb_df['icao24'], adsb_df['local time'], adsb_df['geoaltitude'], adsb_df['velocity'], adsb_df['heading'], adsb_df['vertrate'], adsb_df['TYPE AIRCRAFT'], adsb_df['MODEL']])
# fig2 = px.scatter_3d(adsb_df, x="lon", y="lat", z="geoaltitude")
# fig2.update_traces(marker_size = 5, opacity=0.2)
fig2.update_traces(marker_size = 2, mode="markers+lines")

fig3 = px.scatter_3d(rotorcraft_df, x="lon", y="lat", z="geoaltitude", title='rotorcraft only',color="icao24", hover_data=[rotorcraft_df['icao24'], rotorcraft_df['local time'], rotorcraft_df['geoaltitude'], rotorcraft_df['velocity'], rotorcraft_df['heading'], rotorcraft_df['vertrate'], rotorcraft_df['TYPE AIRCRAFT'], rotorcraft_df['MODEL']])
fig3.update_traces(marker_size = 2, mode="markers+lines")

fig4 = px.scatter_3d(fws_df, x="lon", y="lat", z="geoaltitude", title='fixed wing single engine only',color="icao24", hover_data=[fws_df['icao24'], fws_df['local time'], fws_df['geoaltitude'], fws_df['velocity'], fws_df['heading'], fws_df['vertrate'], fws_df['TYPE AIRCRAFT'], fws_df['MODEL']])
fig4.update_traces(marker_size = 2, mode="markers+lines")

fig5 = px.scatter_3d(fwm_df, x="lon", y="lat", z="geoaltitude", title='fixed wing multi engine only',color="icao24", hover_data=[fwm_df['icao24'], fwm_df['local time'], fwm_df['geoaltitude'], fwm_df['velocity'], fwm_df['heading'], fwm_df['vertrate'], fwm_df['TYPE AIRCRAFT'], fwm_df['MODEL']])
fig5.update_traces(marker_size = 2, mode="markers+lines")


fig1.write_html(case_study_path+"2D_map_all_tracks.html")
fig2.write_html(case_study_path+"3D_map_all_tracks.html")
fig3.write_html(case_study_path+"3D_map_rotorcraft_only.html")
fig4.write_html(case_study_path+"3D_map_fws_only.html")
fig5.write_html(case_study_path+"3D_map_fwm_only.html")


# fig12 = px.scatter_mapbox(adsb_df, lat="lat", lon="lon", color="icao24", hover_data=[adsb_df['icao24'], adsb_df['local time'], adsb_df['geoaltitude'], adsb_df['velocity'], adsb_df['heading'], adsb_df['vertrate'], adsb_df['TYPE AIRCRAFT'], adsb_df['MODEL']])
# fig12.update_traces(marker_size = 2, mode="lines")

fig13 = px.scatter_3d(adsb_df, x="lon", y="lat", z="geoaltitude", color="icao24", hover_data=[adsb_df['track'], adsb_df['icao24'], adsb_df['local time'], adsb_df['geoaltitude'], adsb_df['velocity'], adsb_df['heading'], adsb_df['vertrate'], adsb_df['TYPE AIRCRAFT'], adsb_df['MODEL']])
# fig13 = px.scatter_3d(adsb_df, x="lon", y="lat", z="geoaltitude", color="track", hover_data=[adsb_df['icao24'], adsb_df['local time'], adsb_df['geoaltitude'], adsb_df['velocity'], adsb_df['heading'], adsb_df['vertrate'], adsb_df['TYPE AIRCRAFT'], adsb_df['MODEL']])
# fig2 = px.scatter_3d(adsb_df, x="lon", y="lat", z="geoaltitude")
# fig2.update_traces(marker_size = 5, opacity=0.2)
fig13.update_traces(marker_size = 2, mode="markers+lines")

# fig14 = px.scatter_3d(rotorcraft_df, x="lon", y="lat", z="geoaltitude", title='rotorcraft only',color="icao24", hover_data=[rotorcraft_df['icao24'], rotorcraft_df['local time'], rotorcraft_df['geoaltitude'], rotorcraft_df['velocity'], rotorcraft_df['heading'], rotorcraft_df['vertrate'], rotorcraft_df['TYPE AIRCRAFT'], rotorcraft_df['MODEL']])
# fig14.update_traces(marker_size = 2, mode="markers+lines")
#
# fig15 = px.scatter_3d(fws_df, x="lon", y="lat", z="geoaltitude", title='fixed wing single engine only',color="icao24", hover_data=[fws_df['icao24'], fws_df['local time'], fws_df['geoaltitude'], fws_df['velocity'], fws_df['heading'], fws_df['vertrate'], fws_df['TYPE AIRCRAFT'], fws_df['MODEL']])
# fig15.update_traces(marker_size = 2, mode="markers+lines")
#
# fig16 = px.scatter_3d(fwm_df, x="lon", y="lat", z="geoaltitude", title='fixed wing multi engine only',color="icao24", hover_data=[fwm_df['icao24'], fwm_df['local time'], fwm_df['geoaltitude'], fwm_df['velocity'], fwm_df['heading'], fwm_df['vertrate'], fwm_df['TYPE AIRCRAFT'], fwm_df['MODEL']])
# fig16.update_traces(marker_size = 2, mode="markers+lines")

# fig12.write_html(paper_dir+"/flight tracks/"+"2D_map_all_tracks.html")
fig13.write_html(paper_dir+"/flight tracks/"+"3D_map_all_tracks.html")
# fig14.write_html(paper_dir+"/flight tracks/"+"3D_map_rotorcraft_only.html")
# fig15.write_html(paper_dir+"/flight tracks/"+"3D_map_fws_only.html")
# fig16.write_html(paper_dir+"/flight tracks/"+"3D_map_fwm_only.html")

# fig13.show()

def PlotAircraftTimeSeriesCaseStudyTrack(data, ac_id, path):

    # temp_data = data[data['icao24'] == 'A04E60']
    temp_data = data[data['icao24'] == ac_id]
    # print('temp data =\n', temp_data)
    unique_tracks = temp_data['track'].unique()
    # print('unique track =\n', unique_tracks)

    for track in unique_tracks:
        # print('track = ', track)
        temp_data2 = temp_data[temp_data['track'] == track]
        # print('temp_data2 =\n', temp_data2)

        fig6 = plt.figure()
        fig7 = plt.figure()
        fig8 = plt.figure()
        fig9 = plt.figure()
        fig10 = plt.figure()
        fig11 = plt.figure()
        ax6 = fig6.add_subplot(111)
        ax7 = fig7.add_subplot(111)
        ax8 = fig8.add_subplot(111)
        ax9 = fig9.add_subplot(111)
        ax10 = fig10.add_subplot(111)
        ax11 = fig11.add_subplot(111)

        ax6.plot(temp_data2['datetime'], temp_data2['lat'], label='lat')
        ax6.legend(loc='best')
        # ax6.set_xticklabels(ax6.get_xticks(), rotation=45)

        ax7.plot(temp_data2['datetime'], temp_data2['lon'], label='lon')
        ax7.legend(loc='best')

        ax8.plot(temp_data2['datetime'], temp_data2['geoaltitude'], label='geoaltitude')
        ax8.legend(loc='best')

        ax9.plot(temp_data2['datetime'], temp_data2['velocity'], label='velocity')
        ax9.legend(loc='best')

        ax10.plot(temp_data2['datetime'], temp_data2['heading'], label='heading')
        ax10.legend(loc='best')

        ax11.plot(temp_data2['datetime'], temp_data2['vertrate'], label='vertrate')
        ax11.legend(loc='best')

        # plt.xticks(temp_data2['local time'].datetime())
        # fig6.autofmt_xdate()
        # plt.locator_params(axis='x', nbins=4, tight=True)
        # plt.subplots_adjust(left=0, bottom=3, right=1, top=4, wspace=0, hspace=0)

        fig6.autofmt_xdate()
        fig7.autofmt_xdate()
        fig8.autofmt_xdate()
        fig9.autofmt_xdate()
        fig10.autofmt_xdate()
        fig11.autofmt_xdate()

        # fig_name6 = path+"lat_"+str(ac_id)+"_track_"+str(track)+".png"
        fig_name6 = path+"lat_"+str(ac_id)+"_"+str(track)+".png"
        fig6.savefig(fig_name6)
        fig6.clear()
        plt.close(fig6)

        fig_name7 = path+"lon_"+str(ac_id)+"_"+str(track)+".png"
        fig7.savefig(fig_name7)
        fig7.clear()
        plt.close(fig7)

        fig_name8 = path+"alt_"+str(ac_id)+"_"+str(track)+".png"
        fig8.savefig(fig_name8)
        fig8.clear()
        plt.close(fig8)

        fig_name9 = path+"velocity_"+str(ac_id)+"_"+str(track)+".png"
        fig9.savefig(fig_name9)
        fig9.clear()
        plt.close(fig9)

        fig_name10 = path+"heading_"+str(ac_id)+"_"+str(track)+".png"
        fig10.savefig(fig_name10)
        fig10.clear()
        plt.close(fig10)

        fig_name11 = path+"vertrate_"+str(ac_id)+"_"+str(track)+".png"
        fig11.savefig(fig_name11)
        fig11.clear()
        plt.close(fig11)

# ---------------------------------------------
# USE TQDM LOOP BELOW TO GENERATE INDIVUAL PLOTS FOR EACH SPLIT TRACKS
# (already have them saved inC:\Users\nicol\Google Drive\PhD\Conferences & Papers\AIAA Aviation 2023\Code and results\2_2. Aircraft behaviour category detection\flight tracks\all smooth tracks)
# for icao in tqdm(icao_list):
#     PlotAircraftTimeSeriesCaseStudyTrack(data=adsb_df, ac_id=icao, path=case_study_path)
# ---------------------------------------------


def PlotNClusters(data, path, track_id, ax=None, plt_kwargs={}):

    track_id_list = pd.unique(adsb_df['track_id'])
    print('track_id_list = ', track_id_list)
    print('data = ', data)

    if ax is None:
        fig = plt.gcf()
        ax = plt.gca()

    # ax.plot(data['datetime'], data['geoaltitude'], **plt_kwargs, label=str(track_id_list[track_idx]))

    temp_data = data.loc[data['track_id'] == track_id, 'geoaltitude']
    ax.plot(temp_data.index, temp_data, **plt_kwargs, label=str(track_id))
    ax.scatter(temp_data.index, temp_data, s=3, **plt_kwargs)


    # temp_data_savgol = temp_data
    # temp_data_savgol['geoaltitude'] = savgol_filter(temp_data_savgol['geoaltitude'], window_length=11, polyorder=3, mode="nearest")
    temp_data_savgol = savgol_filter(temp_data, window_length=21, polyorder=3, mode="nearest")
    # ax.plot(temp_data_savgol.index, temp_data_savgol, **plt_kwargs)
    # ax.plot(range(0, len(temp_data_savgol)), temp_data_savgol, **plt_kwargs, label=str(track_id)+'_savgol')
    ax.plot(temp_data.index, temp_data_savgol, **plt_kwargs, label=str(track_id)+'_savgol')
    ax.scatter(temp_data.index, temp_data_savgol, s=3, **plt_kwargs)

    # temp_data_kf = savgol_filter(temp_data, window_length=21, polyorder=3, mode="nearest")
    # ax.plot(temp_data.index, temp_data_kf, **plt_kwargs, label=str(track_id)+'_kf')
    # ax.scatter(temp_data.index, temp_data_kf, s=3, **plt_kwargs)

    # for track_idx in track_id_list:
    #     print('track_idx = ', track_idx)
    #     print(data.loc[data['track_id'] == track_idx, 'geoaltitude'])
    #     # ax.plot(data['datetime'], data[data['track_id'] == track_idx], **plt_kwargs, label=str(track_id_list[track_idx]))
    #     temp_data = data.loc[data['track_id'] == track_idx, 'geoaltitude']
    #     ax.plot(temp_data.index, temp_data, **plt_kwargs)
    #     # ax.plot(temp_data.index, temp_data, **plt_kwargs, label=str(track_id_list[track_idx]))
    #     # ax.scatter(x_data_array, data[icao_idx], **plt_kwargs, label=str(icao_array[icao_idx]))
    ax.legend(loc='best')

    fig_name = path + "track_idx_" + str(track_id) + ".png"
    # fig_name = path + "track_idx_" + ".png"
    fig.savefig(fig_name)
    fig.clear(ax)
    plt.close(fig)
    # return ax

for track_id in pd.unique(adsb_df['track_id']):
    PlotNClusters(data=adsb_df, path=case_study_path, track_id=track_id)



print('smooth adsb_df =\n', adsb_df)
print('smooth adsb_df info =\n', adsb_df.info())

adsb_df = adsb_df.set_axis(adsb_df['datetime'], axis=0)
adsb_df = adsb_df.drop(['time', 'local time'], axis=1)
print('change index to datetime objects and drop time and local time cols')
print('smooth adsb_df 1 =\n', adsb_df)
print('smooth adsb_df 1 info =\n', adsb_df.info())

# adsb_df = adsb_df.col_name.resample('M').mean()
# adsb_df = adsb_df.resample('1Min').mean()
# adsb_df = adsb_df.groupby('icao24').resample('1Min').mean()
# adsb_df = adsb_df.groupby('track_id').resample('1Min').mean()
# adsb_df = adsb_df.groupby('track_id').resample('30S').mean()
adsb_df = adsb_df.groupby('track_id').resample('10S').mean()
# adsb_df = adsb_df.groupby([pd.Grouper(freq='1Min'), 'icao24'])
# adsb_df['Event'].count().unstack()
# adsb_df = adsb_df.unstack('icao24', fill_value=999.99)
# adsb_df = adsb_df.unstack('icao24')
adsb_df = adsb_df.unstack('track_id')

print('rescale datetime objects for average of every 1min intervals')
print('smooth adsb_df 2 =\n', adsb_df)
print('smooth adsb_df 2 info =\n', adsb_df.info())

# print('test =\n', adsb_df[adsb_df['lat', 'A343C4_2'].notnull()])

def SplitVariablesIntoDfs(var, df):
    temp_df = pd.DataFrame()
    temp_df_non_nan = pd.DataFrame()
    for col in df.columns:
        # print('----------------')
        # print('col = ', col)
        # print('col dataframe =\n', df[col])
        if var in col:
            # print('!!! var is in col !!!')
            # print('----------------------')
            # print('col = ', col)
            # print('df[col] =\n', df[col])
            # print('df[col].dropna() =\n', df[col].dropna())

            x = df[col].dropna().set_axis(range(0, len(df[col].dropna())))
            # print('test_series =\n ', x)


            # temp_df = temp_df.append(df[col])
            # temp_df = temp_df.append(df[col].dropna())
            temp_df = temp_df.append(x)
            # print('temp_df =\n', temp_df)


            # print('df[col] =\n', df[col])
            # print('df[col].notnull() =\n', df[col][df[col].notnull()])

            # temp_df2 = pd.DataFrame(df[col][df[col].notnull()])
            # temp_df2.insert(0, 'idx', range(0, len(temp_df2)))
            # temp_df2.set_index('idx')
            # print('temp_df2 =\n', temp_df2)

            # temp_df = temp_df.append(temp_df2)


            # print('shift =\n', df[col].shift(1).ffill())
            # print(df[col].isna().idxmax(1).where(df[col].isna().any(1)))

            # df.shift(periods=3)
            # df["col2_lag"] = df["col2"].shift(1).ffill()

            # # temp_df_non_nan = temp_df_non_nan.append(df[df[col].notnull()])
            # temp_df_non_nan = df[df[col].notnull()]
            #
            # len_idx = len(temp_df_non_nan)
            # print('len_idx = ', len_idx)
            # temp_df_non_nan = temp_df_non_nan.insert(0, 'New_ID', range(0, len(temp_df_non_nan)))
            #
            # # temp_df_non_nan = df[df[col].notnull()]
            # print('temp_df_non_nan =\n', temp_df_non_nan)

    # final_df = temp_df
    final_df = temp_df.T
    # print('final_df = \n', final_df)

    # print('cols = ', list(final_df.columns))
    new_cols = [i[1] for i in list(final_df.columns)]
    # print('new_cols = ', new_cols)
    final_df.columns = new_cols

    return final_df

lat_df = SplitVariablesIntoDfs(var='lat', df=adsb_df)
lon_df = SplitVariablesIntoDfs(var='lon', df=adsb_df)
alt_df = SplitVariablesIntoDfs(var='geoaltitude', df=adsb_df)
vel_df = SplitVariablesIntoDfs(var='velocity', df=adsb_df)
vrate_df = SplitVariablesIntoDfs(var='vertrate', df=adsb_df)
hdg_df = SplitVariablesIntoDfs(var='heading', df=adsb_df)
accel_df = SplitVariablesIntoDfs(var='acceleration', df=adsb_df)
turnrate_df = SplitVariablesIntoDfs(var='turnrate', df=adsb_df)
print('lat_df =\n', lat_df)
# print('\n', lat_df.info())

# print('diff = \n', alt_df.diff())

print('adsb_df =\n', adsb_df)
print('\n', adsb_df.info())
adsb_df.to_csv(case_study_path+'smooth_tracks_dataset_all.csv')

lat_df.to_csv(case_study_path+'smooth_tracks_dataset_lat.csv')
lon_df.to_csv(case_study_path+'smooth_tracks_dataset_lon.csv')
alt_df.to_csv(case_study_path+'smooth_tracks_dataset_alt.csv')
vel_df.to_csv(case_study_path+'smooth_tracks_dataset_vel.csv')
vrate_df.to_csv(case_study_path+'smooth_tracks_dataset_vrate.csv')
hdg_df.to_csv(case_study_path+'smooth_tracks_dataset_hdg.csv')
accel_df.to_csv(case_study_path+'smooth_tracks_dataset_accel.csv')
turnrate_df.to_csv(case_study_path+'smooth_tracks_dataset_turnrate.csv')




