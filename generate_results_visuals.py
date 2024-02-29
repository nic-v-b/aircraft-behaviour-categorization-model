
# https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.colors
from matplotlib.dates import DateFormatter
import plotly.express as px
px.set_mapbox_access_token('pk.eyJ1IjoibmljdmIiLCJhIjoiY2thNzBxMnl0MDAyYzJ0bmZpeW1jOHNlayJ9.p5h0jJ78qIUWcRLQ19muYw')
import sktime
from time import sleep
from tqdm import tqdm
import datetime
from sklearn.cluster import DBSCAN
import matplotlib.patheffects as pe
import seaborn as sns

# set pandas dataframe display properties
max_rows = 50
# max_rows = 875
max_cols = 15
# max_cols = None
pd.set_option('display.width', 320)
pd.set_option('display.max_columns', max_cols)
pd.set_option('display.min_rows', max_rows)
pd.set_option('display.max_rows', max_rows)

data_dir = r'C:/Users/nicol/Google Drive/PhD/Conferences & Papers/AIAA Journal/Code and results/'

# results_dir = data_dir+'results/1 day results/2 clusters/results.csv'
# results_dir = data_dir+'results/1 day results/4 clusters/results.csv'
# results_dir = data_dir+'results/1 day results/10 clusters/results.csv'
# results_dir = data_dir+'results/1 day results/15 clusters/results.csv'
# results_dir = data_dir+'results/1 day results/20 clusters/results.csv'
# results_dir = data_dir+'results/1 day results/25 clusters/results.csv'
# results_dir = data_dir+'results/1 day results/30 clusters/results.csv'

# title_str = '30 clusters_multidim (lon, lat, alt, vel, hdg, vrate) KMeans DTW'
# title_str = '30 clusters_multidim (lon, lat, alt, vel, hdg, vrate) KShape'
#
# title_str = '30 clusters_multidim (lon, lat, alt) KMeans DTW'
# title_str = '30 clusters_multidim (lon, lat, alt) KShape'
#
# title_str = '30 clusters_multidim (vel, hdg, vrate) KMeans DTW'
# title_str = '30 clusters_multidim (vel, hdg, vrate) KShape'
#
# title_str = '30 clusters_multidim (lon, lat, alt, vrate) KMeans DTW'
# # title_str = '30 clusters_multidim (lon, lat, alt, vrate) KShape'

# 30 clusters_multidim (lon, lat, alt, vel, turnrate, acceleration, vrate) KMeans DTW

# title_str = '30 clusters_multidim (lon, lat, alt, vel, turnrate, acceleration, vrate) KMeans DTW'
# title_str = '30 clusters_multidim (lon, lat, alt, turnrate, acceleration, vrate) KMeans DTW'
# title_str = '30 clusters_multidim (vel, turnrate, acceleration, vrate) KMeans DTW'
# title_str = '30 clusters_multidim (turnrate, acceleration, vrate) KMeans DTW'

# n_clusters = 2
# n_clusters = 3
n_clusters = 4
# n_clusters = 30
title_str = str(n_clusters)

from matplotlib.pyplot import cm
color_list = cm.Set3(np.linspace(0, 1, n_clusters))

# results_dir = data_dir+'results/1 day results/'+title_str+'/results.csv'
# results_dir = data_dir+'results/nb clusters '+str(n_clusters)+'_results.csv'
# results_dir = data_dir+'results/K=2, 3, 4, 30 for lon lat alt/nb clusters '+str(n_clusters)+'_results.csv'
# results_dir = data_dir+'results/K=4/run 4/nb clusters '+str(n_clusters)+'_results.csv'
# results_dir = data_dir+'results/K=4/run 6/nb clusters '+str(n_clusters)+'_results.csv'
results_dir = data_dir+'results/K=4/current run/nb clusters '+str(n_clusters)+'_results.csv'

# title_str = 'nbclusters_30_results'
# results_dir = data_dir+'results/1 day results/'+title_str+'.csv'


cluster_data_df = pd.read_csv(results_dir)
cluster_data_df = cluster_data_df.loc[:, ~cluster_data_df.columns.str.contains('^Unnamed')]
# data_df = data_df.fillna('filler')
print('cluster_data_df = \n', cluster_data_df)

flight_tracks_dir = r'C:/Users/nicol/Google Drive/PhD/Conferences & Papers/AIAA Aviation 2023/Code and results/2_2. Aircraft behaviour category detection/flight tracks/'

ac_info_df = pd.read_csv(flight_tracks_dir+'final data 1 day/smooth_tracks_dataset.csv')
ac_info_df = ac_info_df.loc[:, ~ac_info_df.columns.str.contains('^Unnamed')]
ac_info_df = ac_info_df.drop("track", axis='columns')
# ac_info_df = ac_info_df.drop("heading", axis='columns')
# ac_info_df = ac_info_df[ac_info_df['icao24'] != 'A54E6A']
# ac_info_df = ac_info_df[ac_info_df['icao24'] != 'A69939']
# ac_info_df = ac_info_df[ac_info_df['icao24'] != 'A44416']
# ac_info_df = ac_info_df[ac_info_df['icao24'] != 'A19A48']
# ac_info_df = ac_info_df[ac_info_df['icao24'] != 'A369AF']
# ac_info_df = ac_info_df[ac_info_df['icao24'] != 'A71B82']
# ac_info_df = ac_info_df[ac_info_df['icao24'] != 'AB9AB3']
# ac_info_df = ac_info_df[ac_info_df['icao24'] != 'A32ED8']
# ac_info_df = ac_info_df[ac_info_df['icao24'] != 'ADD25E']
# ac_info_df = ac_info_df[ac_info_df['icao24'] != 'A54EB1']

# ac_info_df = ac_info_df[ac_info_df['geoaltitude'] > 9000]
# ac_info_df = ac_info_df[ac_info_df['icao24'] != 'A0011C']

# ac_info_df = ac_info_df.drop(['velocity', 'turnrate', 'acceleration', 'vertrate'], axis='columns')
# ac_info_df = ac_info_df.drop(['lon', 'lat', 'geoaltitude', 'velocity'], axis='columns')
# ac_info_df = ac_info_df.drop(['lon', 'lat', 'geoaltitude'], axis='columns')

# ac_info_df = ac_info_df[(ac_info_df['turnrate'] > -5) & (ac_info_df['turnrate'] < 5)]
# ac_info_df = ac_info_df[(ac_info_df['acceleration'] > -5) & (ac_info_df['acceleration'] < 5)]

# ac_info_df = ac_info_df[:50]  # for debugging
# ac_info_df = ac_info_df[:1000]  # for debugging
# print('ac_info_df = \n', ac_info_df)

# centroids_lon_df = pd.read_csv(data_dir+'results/K=4/run 6/nb clusters '+str(n_clusters)+'_lon_centroids.csv')
# centroids_lat_df = pd.read_csv(data_dir+'results/K=4/run 6/nb clusters '+str(n_clusters)+'_lat_centroids.csv')
# centroids_alt_df = pd.read_csv(data_dir+'results/K=4/run 6/nb clusters '+str(n_clusters)+'_alt_centroids.csv')
centroids_lon_df = pd.read_csv(data_dir+'results/K=4/current run/nb clusters '+str(n_clusters)+'_lon_centroids.csv')
centroids_lat_df = pd.read_csv(data_dir+'results/K=4/current run/nb clusters '+str(n_clusters)+'_lat_centroids.csv')
centroids_alt_df = pd.read_csv(data_dir+'results/K=4/current run/nb clusters '+str(n_clusters)+'_alt_centroids.csv')
centroids_lon_df = centroids_lon_df.loc[:, ~centroids_lon_df.columns.str.contains('^Unnamed')]
centroids_lat_df = centroids_lat_df.loc[:, ~centroids_lat_df.columns.str.contains('^Unnamed')]
centroids_alt_df = centroids_alt_df.loc[:, ~centroids_alt_df.columns.str.contains('^Unnamed')]
# print('centroids_lon_df = \n', centroids_lon_df)
# print('centroids_lat_df = \n', centroids_lat_df)
# print('centroids_alt_df = \n', centroids_alt_df)

def plot_2D_centroids_all_clusters(lon_df, lat_df, ac_info_df, path, ax=None, plt_kwargs={}):

    # n_clusters = len(pd.unique(df['cluster']))
    n_clusters = len(lon_df. columns)
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

    fig_name = path + "2D_centroids_map.png"
    # plt.show()
    fig.savefig(fig_name, dpi=200)

def interpret_results_WCSS_Minimizers(features, lon_centroids, lat_centroids, alt_centroids, path, ax=None, plt_kwargs={}):
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
    temp_centroids = [[0] * len(features) for i in range(len(lon_centroids))]
    # print('initialization of temp_centroids = ', temp_centroids)

    for cluster_idx in range(0, len(lon_centroids)):
        # print('cluster_idx = ', cluster_idx)
        # print('centroids[cluster_idx] =\n', centroids[cluster_idx])

        for feature_idx, feature in enumerate(lon_centroids[cluster_idx]):
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
    rows = features

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


# print('max ac_info_df[NO-SEATS]', max(ac_info_df['NO-SEATS']))
# print('min ac_info_df[NO-SEATS]', min(ac_info_df['NO-SEATS']))
#
# print('max ac_info_df[NO-ENG]', max(ac_info_df['NO-ENG']))
# print('min ac_info_df[NO-ENG]', min(ac_info_df['NO-ENG']))

# unique_counts_df = pd.DataFrame()
# temp_df = pd.DataFrame({'TYPE AIRCRAFT': ac_info_df['TYPE AIRCRAFT'].value_counts()})
# unique_counts_df = pd.concat([unique_counts_df, temp_df], axis=1)
# temp_df = pd.DataFrame({'CERTIFICATION': ac_info_df['CERTIFICATION'].value_counts()})
# unique_counts_df = pd.concat([unique_counts_df, temp_df], axis=1)
# temp_df = pd.DataFrame({'TYPE ENGINE': ac_info_df['TYPE ENGINE'].value_counts()})
# unique_counts_df = pd.concat([unique_counts_df, temp_df], axis=1)
# temp_df = pd.DataFrame({'MODEL': ac_info_df['MODEL'].value_counts()})
# unique_counts_df = pd.concat([unique_counts_df, temp_df], axis=1)
# temp_df = pd.DataFrame({'NO-ENG': ac_info_df['NO-ENG'].value_counts()})
# unique_counts_df = pd.concat([unique_counts_df, temp_df], axis=1)
# temp_df = pd.DataFrame({'NO-SEATS': ac_info_df['NO-SEATS'].value_counts()})
# unique_counts_df = pd.concat([unique_counts_df, temp_df], axis=1)
# temp_df = pd.DataFrame({'AC-WEIGHT': ac_info_df['AC-WEIGHT'].value_counts()})
# unique_counts_df = pd.concat([unique_counts_df, temp_df], axis=1)
# print(ac_info_df.groupby(['track_id','TYPE AIRCRAFT']).size().unstack(fill_value=0))
# print(ac_info_df.groupby(['icao24','TYPE AIRCRAFT']).size().unstack(fill_value=0))

unique_counts_df = pd.DataFrame()
# print(ac_info_df.groupby('track_id')['TYPE AIRCRAFT'].nunique())
# print(ac_info_df.groupby('icao24')['TYPE AIRCRAFT'].nunique())
# print(ac_info_df.groupby('TYPE AIRCRAFT')['icao24'].nunique())
temp_df = pd.DataFrame({'TYPE AIRCRAFT': ac_info_df.groupby('TYPE AIRCRAFT')['icao24'].nunique()})
unique_counts_df = pd.concat([unique_counts_df, temp_df], axis=1)
temp_df = pd.DataFrame({'CERTIFICATION': ac_info_df.groupby('CERTIFICATION')['icao24'].nunique()})
unique_counts_df = pd.concat([unique_counts_df, temp_df], axis=1)
temp_df = pd.DataFrame({'TYPE ENGINE': ac_info_df.groupby('TYPE ENGINE')['icao24'].nunique()})
unique_counts_df = pd.concat([unique_counts_df, temp_df], axis=1)
temp_df = pd.DataFrame({'MODEL': ac_info_df.groupby('MODEL')['icao24'].nunique()})
unique_counts_df = pd.concat([unique_counts_df, temp_df], axis=1)
temp_df = pd.DataFrame({'NO-ENG': ac_info_df.groupby('NO-ENG')['icao24'].nunique()})
unique_counts_df = pd.concat([unique_counts_df, temp_df], axis=1)
temp_df = pd.DataFrame({'NO-SEATS': ac_info_df.groupby('NO-SEATS')['icao24'].nunique()})
unique_counts_df = pd.concat([unique_counts_df, temp_df], axis=1)
temp_df = pd.DataFrame({'AC-WEIGHT': ac_info_df.groupby('AC-WEIGHT')['icao24'].nunique()})
unique_counts_df = pd.concat([unique_counts_df, temp_df], axis=1)

# unique_counts_df = pd.DataFrame()
# # temp_df = pd.DataFrame({'TYPE AIRCRAFT': ac_info_df.groupby(['track_id','TYPE AIRCRAFT']).size().unstack(fill_value=0)})
# # temp_df = ac_info_df.groupby(['track_id','TYPE AIRCRAFT']).size().unstack(fill_value=0)
# temp_df = ac_info_df.groupby(['icao24','TYPE AIRCRAFT']).size().unstack(fill_value=0)
# unique_counts_df = pd.concat([unique_counts_df, temp_df], axis=1)
# unique_counts_df = unique_counts_df.sort_values(by='Fixed wing multi engine', ascending=False)
# unique_counts_df = unique_counts_df.sort_values(by='Fixed wing single engine', ascending=False)
# unique_counts_df = unique_counts_df.sort_values(by='Rotorcraft', ascending=False)
# df.sort_values(by=['col1', 'col2'])

# # temp_df = pd.DataFrame({'CERTIFICATION': ac_info_df['CERTIFICATION'].value_counts()})
# temp_df = ac_info_df.groupby(['icao24','CERTIFICATION']).size().unstack(fill_value=0)
# unique_counts_df = pd.concat([unique_counts_df, temp_df], axis=1)

# temp_df = pd.DataFrame({'TYPE ENGINE': ac_info_df['TYPE ENGINE'].value_counts()})
# unique_counts_df = pd.concat([unique_counts_df, temp_df], axis=1)
# temp_df = pd.DataFrame({'MODEL': ac_info_df['MODEL'].value_counts()})
# unique_counts_df = pd.concat([unique_counts_df, temp_df], axis=1)
# temp_df = pd.DataFrame({'NO-ENG': ac_info_df['NO-ENG'].value_counts()})
# unique_counts_df = pd.concat([unique_counts_df, temp_df], axis=1)
# temp_df = pd.DataFrame({'NO-SEATS': ac_info_df['NO-SEATS'].value_counts()})
# unique_counts_df = pd.concat([unique_counts_df, temp_df], axis=1)
# temp_df = pd.DataFrame({'AC-WEIGHT': ac_info_df['AC-WEIGHT'].value_counts()})
# unique_counts_df = pd.concat([unique_counts_df, temp_df], axis=1)

print('unique_counts_df = \n', unique_counts_df)
# unique_counts_df.to_csv(data_dir+"results/unique_counts_total_df.csv")
# unique_counts_df.to_csv(data_dir+"results/unique_counts_total_df.csv")
unique_counts_df.to_csv(data_dir+"results/K=2, 3, 4, 30 for lon lat alt/unique_counts_total_df.csv")

unique_track_ids_list = []
for col in cluster_data_df.columns:
    temp = list(cluster_data_df[col].unique())
    # print('temp = \n', temp)
    unique_track_ids_list = unique_track_ids_list + temp
# print('unique_track_ids_list = \n', unique_track_ids_list)

ac_info_df['cluster'] = 0
# ac_info_df.loc[ac_info_df.track_id == 'AA9A0A_1', 'cluster'] = "cluster_0"
for unique_track in unique_track_ids_list:
    for col in cluster_data_df.columns:
        if (unique_track in cluster_data_df[col].unique()) == True:
            # print('---------------')
            # print(unique_track+' is in '+str(col))
            ac_info_df.loc[ac_info_df.track_id == unique_track, 'cluster'] = str(col)

ac_info_df = ac_info_df[ac_info_df.cluster != 0]

# ac_info_df['cluster'] = np.where(adsb_df['track'] == data_df)
print('ac_info_df = \n', ac_info_df)

# normal_cat_ac_df = ac_info_df
# # keep only ac below 19000 lbs
# normal_cat_ac_df = normal_cat_ac_df[(normal_cat_ac_df['AC-WEIGHT'] == '12,500 - 19,999') | (normal_cat_ac_df['AC-WEIGHT'] == 'Up to 12,499')]
# # keep only ac with PAX <= 19 seats
# normal_cat_ac_df = normal_cat_ac_df[normal_cat_ac_df['NO-SEATS'] <= 19]
# # keep only ac with speeds <= 128 m/s (250KCAS)
# # normal_cat_ac_df = normal_cat_ac_df[normal_cat_ac_df['velocity'] <= 128]
# # normal_cat_ac_df = normal_cat_ac_df[(normal_cat_ac_df['NO-SEATS'] <= 19) & (normal_cat_ac_df['turnrate'] < 5)]
# print('normal_cat_ac_df = \n', normal_cat_ac_df)
#
# normal_cat_ac_df['perf level'] = 0
# normal_cat_ac_df['perf level'] = np.where(normal_cat_ac_df['velocity'] > 128, 'high speed', normal_cat_ac_df['perf level'])
# normal_cat_ac_df['perf level'] = np.where(normal_cat_ac_df['velocity'] < 128, 'low speed', normal_cat_ac_df['perf level'])
#
# normal_cat_ac_df['cert level'] = 0
# normal_cat_ac_df['cert level'] = np.where(normal_cat_ac_df['NO-SEATS'].between(0, 1), 'level 1', normal_cat_ac_df['cert level'])
# normal_cat_ac_df['cert level'] = np.where(normal_cat_ac_df['NO-SEATS'].between(2, 6), 'level 2', normal_cat_ac_df['cert level'])
# normal_cat_ac_df['cert level'] = np.where(normal_cat_ac_df['NO-SEATS'].between(7, 9), 'level 3', normal_cat_ac_df['cert level'])
# normal_cat_ac_df['cert level'] = np.where(normal_cat_ac_df['NO-SEATS'].between(10, 19), 'level 4', normal_cat_ac_df['cert level'])
#
# print('normal_cat_ac_df = \n', normal_cat_ac_df)
# # normal_cat_ac_df.to_csv(results_dir+"normal_cat_ac_df.csv")
# # normal_cat_ac_df.to_csv(data_dir+"results/normal_cat_ac_df.csv")
# normal_cat_ac_df.to_csv(data_dir+"results/K=2, 3, 4, 30 for lon lat alt/normal_cat_ac_df.csv")
#
# all_clusters_certification_df = pd.DataFrame()
# all_clusters_typeac_df = pd.DataFrame()
# all_clusters_acweight_df = pd.DataFrame()
# all_clusters_typeeng_df = pd.DataFrame()
# all_clusters_nbseats_df = pd.DataFrame()
# all_clusters_norm_perflvl_df = pd.DataFrame()
# all_clusters_norm_certlvl_df = pd.DataFrame()
# for cluster_idx in list(np.arange(0, len(list(cluster_data_df.columns)), 1)):
#     print('-------------------------')
#     print('cluster_idx = ', cluster_idx)
#     temp_df = ac_info_df[ac_info_df['cluster'] == 'cluster_'+str(cluster_idx)]
#     temp_norm_df = normal_cat_ac_df[normal_cat_ac_df['cluster'] == 'cluster_'+str(cluster_idx)]
#     # print('temp_df = \n', temp_df)
#
#     # if temp_df['CERTIFICATION'] == str(['Experimental', 'Research and Development']):
#     #     print("['Experimental', 'Research and Development'] in cluster ", str(cluster_idx))
#     # if temp_df['CERTIFICATION'] == str(['Experimental', 'To show compliance with FAR']):
#     #     print("['Experimental', 'To show compliance with FAR'] in cluster ", str(cluster_idx))
#     # if temp_df['CERTIFICATION'] == str(['Experimental', 'Research and Development, Operating Kit Built Aircraft']):
#     #     print("['Experimental', 'Research and Development, Operating Kit Built Aircraft'] in cluster ", str(cluster_idx))
#     # if temp_df['CERTIFICATION'] == str(['Standard', 'Acrobatic']):
#     #     print("['Standard', 'Acrobatic'] in cluster ", str(cluster_idx))
#
#     if temp_df['CERTIFICATION'].str.count('Experimental').sum() > 0:
#         print('++++++')
#         print(temp_df.groupby('CERTIFICATION')['track_id'].nunique())
#     # print('number of normal cat = ', temp_df['CERTIFICATION'].str.count('Experimental').sum())
#     # print('number of normal cat = ', temp_df['CERTIFICATION'].str.contains('Experimental', regex=True))
#     # print('number of normal cat = ', temp_df['CERTIFICATION'].str.contains('Normal', regex=True))
#     # print('number of normal cat type = ', type(temp_df['CERTIFICATION'].str.contains('Normal', regex=True)))
#     # try:
#     #     print('number of normal cat = ', temp_df['CERTIFICATION'].str.count("Normal")).sum()
#     # except AttributeError:
#     #     print('number of normal cat = 0')
#     # print(temp_df.groupby('CERTIFICATION')['icao24'].nunique())
#     # print(temp_df.groupby('CERTIFICATION')['icao24'].nunique())
#     # print(temp_df.groupby('icao24')['CERTIFICATION'].count('Normal'))
#     # print(temp_df.groupby('TYPE AIRCRAFT')['track_id'].nunique())
#     # print('sum = ', temp_df.groupby('TYPE AIRCRAFT')['track_id'].nunique().sum())
#     # print(temp_df.groupby('CERTIFICATION')['track_id'].nunique())
#     # print('sum = ', temp_df.groupby('CERTIFICATION')['track_id'].nunique().sum())
#     # print('+++')
#     # print(temp_df.groupby('TYPE AIRCRAFT')['track_id'].nunique())
#     # print('+++')
#     # print(temp_df.groupby('CERTIFICATION')['track_id'].nunique())
#     # temp_all_clusters_certification_df = pd.DataFrame({'cluster_'+str(cluster_idx): temp_df.groupby('CERTIFICATION')['track_id'].nunique()})
#     # all_clusters_certification_df = pd.concat([all_clusters_certification_df, temp_all_clusters_certification_df], axis=1)
#     # print('+++')
#     # print(temp_df.groupby('TYPE ENGINE')['track_id'].nunique())
#     # print('+++')
#     # print(temp_df.groupby('NO-ENG')['track_id'].nunique())
#     # print('+++')
#     # print(temp_df.groupby('AC-WEIGHT')['track_id'].nunique())
#     # print('+++---+++')
#     # print('sum = ', temp_df.groupby('AC-WEIGHT')['track_id'].nunique().sum())
#
#     temp_all_clusters_certification_df = pd.DataFrame({'cluster_'+str(cluster_idx): temp_df.groupby('CERTIFICATION')['track_id'].nunique()})
#     all_clusters_certification_df = pd.concat([all_clusters_certification_df, temp_all_clusters_certification_df], axis=1)
#
#     temp_all_clusters_typeac_df = pd.DataFrame({'cluster_'+str(cluster_idx): temp_df.groupby('TYPE AIRCRAFT')['track_id'].nunique()})
#     all_clusters_typeac_df = pd.concat([all_clusters_typeac_df, temp_all_clusters_typeac_df], axis=1)
#
#     temp_all_clusters_acweight_df = pd.DataFrame({'cluster_'+str(cluster_idx): temp_df.groupby('AC-WEIGHT')['track_id'].nunique()})
#     all_clusters_acweight_df = pd.concat([all_clusters_acweight_df, temp_all_clusters_acweight_df], axis=1)
#
#     temp_all_clusters_typeeng_df = pd.DataFrame({'cluster_'+str(cluster_idx): temp_df.groupby('TYPE ENGINE')['track_id'].nunique()})
#     all_clusters_typeeng_df = pd.concat([all_clusters_typeeng_df, temp_all_clusters_typeeng_df], axis=1)
#
#     temp_all_clusters_nbseats_df = pd.DataFrame({'cluster_'+str(cluster_idx): temp_df.groupby('NO-SEATS')['track_id'].nunique()})
#     all_clusters_nbseats_df = pd.concat([all_clusters_nbseats_df, temp_all_clusters_nbseats_df], axis=1)
#
#     # print('type sum   = ', temp_df.groupby('TYPE AIRCRAFT')['track_id'].nunique().sum())
#     # print('weight sum = ', temp_df.groupby('AC-WEIGHT')['track_id'].nunique().sum())
#     # print('perf sum   = ', temp_norm_df.groupby('perf level')['track_id'].nunique().sum())
#     # print('cert sum   = ', temp_norm_df.groupby('cert level')['track_id'].nunique().sum())
#
#
#     temp_all_clusters_norm_perflvl_df = pd.DataFrame({'cluster_'+str(cluster_idx): temp_norm_df.groupby('perf level')['track_id'].nunique()})
#     all_clusters_norm_perflvl_df = pd.concat([all_clusters_norm_perflvl_df, temp_all_clusters_norm_perflvl_df], axis=1)
#
#     temp_all_clusters_norm_certlvl_df = pd.DataFrame({'cluster_'+str(cluster_idx): temp_norm_df.groupby('cert level')['track_id'].nunique()})
#     all_clusters_norm_certlvl_df = pd.concat([all_clusters_norm_certlvl_df, temp_all_clusters_norm_certlvl_df], axis=1)
#
# all_clusters_certification_df = all_clusters_certification_df.T
# all_clusters_typeac_df = all_clusters_typeac_df.T
# all_clusters_acweight_df = all_clusters_acweight_df.T
# all_clusters_typeeng_df = all_clusters_typeeng_df.T
# all_clusters_nbseats_df = all_clusters_nbseats_df.T
# all_clusters_norm_perflvl_df = all_clusters_norm_perflvl_df.T
# all_clusters_norm_certlvl_df = all_clusters_norm_certlvl_df.T
# # print('all_clusters_certification_df =\n', all_clusters_certification_df)
#
# # all_clusters_certification_df.to_csv(results_dir+"all_clusters_certification_df.csv")
# # all_clusters_certification_df.to_csv(data_dir+"results/all_clusters_certification_df.csv")
# # all_clusters_typeac_df.to_csv(data_dir+"results/all_clusters_typeac_df.csv")
# # all_clusters_acweight_df.to_csv(data_dir+"results/all_clusters_acweight_df.csv")
# # all_clusters_typeeng_df.to_csv(data_dir+"results/all_clusters_typeeng_df.csv")
# # all_clusters_nbseats_df.to_csv(data_dir+"results/all_clusters_nbseats_df.csv")
# # all_clusters_norm_perflvl_df.to_csv(data_dir+"results/all_clusters_norm_perflvl_df.csv")
# # all_clusters_norm_certlvl_df.to_csv(data_dir+"results/all_clusters_norm_certlvl_df.csv")
# all_clusters_certification_df.to_csv(data_dir+"results/K=2, 3, 4, 30 for lon lat alt/all_clusters_certification_df.csv")
# all_clusters_typeac_df.to_csv(data_dir+"results/K=2, 3, 4, 30 for lon lat alt/all_clusters_typeac_df.csv")
# all_clusters_acweight_df.to_csv(data_dir+"results/K=2, 3, 4, 30 for lon lat alt/all_clusters_acweight_df.csv")
# all_clusters_typeeng_df.to_csv(data_dir+"results/K=2, 3, 4, 30 for lon lat alt/all_clusters_typeeng_df.csv")
# all_clusters_nbseats_df.to_csv(data_dir+"results/K=2, 3, 4, 30 for lon lat alt/all_clusters_nbseats_df.csv")
# all_clusters_norm_perflvl_df.to_csv(data_dir+"results/K=2, 3, 4, 30 for lon lat alt/all_clusters_norm_perflvl_df.csv")
# all_clusters_norm_certlvl_df.to_csv(data_dir+"results/K=2, 3, 4, 30 for lon lat alt/all_clusters_norm_certlvl_df.csv")

plot_2D_centroids_all_clusters(lon_df=centroids_lon_df, lat_df=centroids_lat_df, ac_info_df=ac_info_df,path=data_dir+'results/K=4/current run/')
interpret_results_WCSS_Minimizers(features=['lon', 'lat', 'alt'], lon_centroids=centroids_lon_df, lat_centroids=centroids_lat_df, alt_centroids=centroids_alt_df, path=data_dir+'results/K=4/current run/')

# ---------------------------------------------------------------------------------------------------------------------
# 2D and 3D trajectories
# ---------------------------------------------------------------------------------------------------------------------
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

ac_info_df = ac_info_df.rename(columns={'lat':'Latitude [deg]',
                                  'lon':'Longitude [deg]',
                                  'geoaltitude':'Altitude [m]' ,
                                  'velocity':'Velocity [m/s]',
                                  'heading':'Heading [deg]',
                                  'turnrate':'Turn Rate [deg/s]',
                                  'acceleration':'Acceleration [m/s2]',
                                  'vertrate':'Vertical Rate [m/s]'})

# 2D trajectories on a map
# fig1 = px.scatter_mapbox(data_df, lat="lat", lon="lon", color="icao24", hover_data=[adsb_df['icao24'], adsb_df['local time'], adsb_df['geoaltitude'], adsb_df['velocity'], adsb_df['heading'], adsb_df['vertrate'], adsb_df['TYPE AIRCRAFT'], adsb_df['MODEL']])
# fig1 = px.scatter_mapbox(ac_info_df, lat="lat", lon="lon", color="icao24", hover_data=hover_data_list)
# fig1 = px.scatter_mapbox(ac_info_df, lat="lat", lon="lon", color="track_id", hover_data=hover_data_list)
fig1 = px.scatter_mapbox(ac_info_df, lat="Latitude [deg]", lon="Longitude [deg]", color="cluster", title=title_str, hover_data=hover_data_list)
# fig1.update_traces(marker_size=1, mode="lines", opacity=0.9)
fig1.update_traces(marker_size=2, opacity=0.99)
# fig1.show()

# 3D trajectories
# fig2 = px.scatter_3d(ac_info_df, x="lon", y="lat", z="geoaltitude", title='3D trajectories',color="icao24", hover_data=[ac_info_df['icao24'], ac_info_df['local time'], ac_info_df['geoaltitude'], ac_info_df['velocity'], ac_info_df['heading'], ac_info_df['vertrate'], ac_info_df['TYPE AIRCRAFT'], ac_info_df['MODEL']])
fig2 = px.scatter_3d(ac_info_df, x="Longitude [deg]", y="Latitude [deg]", z="Altitude [m]", title=title_str, color="cluster", hover_data=hover_data_list)
# fig2 = px.scatter_3d(ac_info_df, x="Longitude [deg]", y="Latitude [deg]", z="Altitude [m]", title=title_str, color="cluster", hover_data=hover_data_list, marker=dict(color=color_list))
# fig2 = px.scatter_3d(ac_info_df, x="Longitude [deg]", y="Latitude [deg]", z="Altitude [m]", title=title_str, color=dict("cluster", color_list), hover_data=hover_data_list)
# fig2 = px.scatter_3d(ac_info_df, x="Longitude [deg]", y="Latitude [deg]", z="Altitude [m]", title=title_str,color="cluster", hover_data=hover_data_list, range_z=[0, 14000])
# fig2.update_traces(marker_size=2, mode="markers+lines")
fig2.update_traces(marker_size=2, opacity=0.7)
# fig2.update_traces(marker_size=2, opacity=0.7, marker_color=color_list)
# fig2.update_traces(marker_size=2, mode="lines")
# fig2.show()

import plotly.figure_factory as ff

fig3 = px.density_mapbox(ac_info_df, lat="Latitude [deg]", lon="Longitude [deg]", radius=2, title=title_str, hover_data=hover_data_list)
# fig3 = px.density_mapbox(ac_info_df, lat="Latitude [deg]", lon="Longitude [deg]", z="cluster", title=title_str, hover_data=hover_data_list)
# fig3 = ff.create_hexbin_mapbox(ac_info_df, lat="Latitude [deg]", lon="Longitude [deg]", nx_hexagon=50, opacity=0.7)
# fig3.update_traces(marker_size=2, opacity=0.99)
# fig3.show()



# def multivariate_timeseries_plot4(multitimeseries_data, model_preds,centroids, color_list, cluster_idx, path, features, ax=None, plt_kwargs={}):
#     # temp_data = data[data['icao24'] == ac_id]
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
#     # # make a list of random colors for each unique cluster
#     # color_list = get_colors(cluster_idx)
#     # color_list = cm.rainbow(np.linspace(0, 1, cluster_idx))
#     # color_list = cm.Set2(np.linspace(0, 1, cluster_idx))
#     # print('color_list = ', color_list)
#
#     multitimeseries_data = scaler.inverse_transform(multitimeseries_data.reshape(multitimeseries_data.shape[0], -1)).reshape(multitimeseries_data.shape)
#     # multitimeseries_data = scaler.inverse_transform(multitimeseries_data)
#
#     centroids = scaler.inverse_transform(centroids.reshape(centroids.shape[0], -1)).reshape(centroids.shape)
#     # scaler.inverse_transform(centroids)
#     # centroids = scaler.inverse_transform(centroids)
#
#     if features == ['lon', 'lat', 'alt']:
#         fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
#
#     if features == ['lon', 'lat', 'alt', 'hdg']:
#         fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
#
#     for i, cluster in enumerate(np.unique(model_preds)):
#         # print('=================================================')
#         print('     --> plotting cluster', str(i),'(from',str(cluster_idx),'total clusters)')
#         # print('=================================================')
#
#         # print('cluster name = ', i)
#         # cluster_idx = cluster_idx-1
#         # cluster_idx = cluster
#         # print('cluster     = ', cluster_id)
#         # print('cluster     = ', cluster)
#         # print('cluster_idx = ', cluster_idx)
#         # print('model_preds = ', model_preds)
#
#         # print('centroids   = ', centroids)
#         # print('centroids.shape   = ', centroids.shape)
#         # print('centroids[cluster_idx]   = ', centroids)
#         # print('centroids[cluster_idx].shape   = ', centroids[cluster_idx].shape)
#         # alt_centroids = centroids[cluster_idx][0]
#         # lon_centroids = centroids[cluster_idx][1]
#         # lat_centroids = centroids[cluster_idx][2]
#
#         if features == ['lon', 'lat', 'alt']:
#             lon_centroids = centroids[i][0]
#             lat_centroids = centroids[i][1]
#             alt_centroids = centroids[i][2]
#             # lon_centroids = ac_info_data_for_cluster.groupby(['time'])['lon'].mean().tolist()
#             # lat_centroids = ac_info_data_for_cluster.groupby(['time'])['lat'].mean().tolist()
#             # alt_centroids = ac_info_data_for_cluster.groupby(['time'])['geoaltitude'].mean().tolist()
#
#         if features == ['lon', 'lat', 'alt', 'hdg']:
#             lon_centroids = centroids[i][0]
#             lat_centroids = centroids[i][1]
#             alt_centroids = centroids[i][2]
#             hdg_centroids = centroids[i][3]
#         # print('alt_centroids = ', alt_centroids)
#         # print('lon_centroids = ', lon_centroids)
#         # print('lat_centroids = ', lat_centroids)
#         # print('alt_centroids.shape = ', alt_centroids.shape)
#         # print('lon_centroids.shape = ', lon_centroids.shape)
#         # print('lat_centroids.shape = ', lat_centroids.shape)
#         # print('     nb in cluster = ', len(lon_centroids))
#
#         # get individual trajectories that are in the cluster i and return their indexes
#         # cluster_icao_idx = np.where(model_preds == cluster_idx)[0]
#         cluster_icao_idx = np.where(model_preds == i)[0]
#         # print('cluster_icao_idx = ', cluster_icao_idx)
#         # track_id_array = np.array(track_id_list)
#         # print('track_id in cluster = ', track_id_array[np.where(y==cluster_idx)[0]])
#         # print('cluster centroid = \n', cluster_centers[cluster_idx])
#         # print('nb of traj in cluster', str(i),'= ', len(cluster_icao_idx))
#
#
#         # if ax is None:
#         #     # fig = plt.gcf().set_size_inches(10, 5)
#         #     fig = plt.gcf()
#         #     # ax = plt.gca()
#         #     # fig, axs = plt.subplots(1, 3)
#         #     # axs = plt.subplots(1, 3)
#         #     fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
#
#         # for icao in icao_array[np.where(y==cluster_idx)[0]]:
#         for icao_idx in cluster_icao_idx:
#             # print('+++    _idx = ', icao_idx)
#             # print('len(cluster_icao_idx) = ', len(cluster_icao_idx))
#             # print('cluster_icao_idx[-1] = ', cluster_icao_idx[-1])
#
#             # print('multitimeseries_data[icao_idx] = ', multitimeseries_data[icao_idx])
#             # print('multitimeseries_data[icao_idx].shape = ', multitimeseries_data[icao_idx].shape)
#             # print('multitimeseries_data[icao_idx][0] = ', multitimeseries_data[icao_idx][0])
#             # print('multitimeseries_data[icao_idx][1] = ', multitimeseries_data[icao_idx][1])
#             # print('multitimeseries_data[icao_idx][2] = ', multitimeseries_data[icao_idx][2])
#             if features == ['lon', 'lat', 'alt']:
#                 lon_data_for_icao_idx = multitimeseries_data[icao_idx][0]
#                 lat_data_for_icao_idx = multitimeseries_data[icao_idx][1]
#                 alt_data_for_icao_idx = multitimeseries_data[icao_idx][2]
#             if features == ['lon', 'lat', 'alt', 'hdg']:
#                 lon_data_for_icao_idx = multitimeseries_data[icao_idx][0]
#                 lat_data_for_icao_idx = multitimeseries_data[icao_idx][1]
#                 alt_data_for_icao_idx = multitimeseries_data[icao_idx][2]
#                 hdg_data_for_icao_idx = multitimeseries_data[icao_idx][3]
#             # print('alt_data_for_icao_idx = ', alt_data_for_icao_idx)
#             # print('alt_data_for_icao_idx.shape = ', alt_data_for_icao_idx.shape)
#             # print('lon_data_for_icao_idx = ', lon_data_for_icao_idx)
#             # print('lon_data_for_icao_idx.shape = ', lon_data_for_icao_idx.shape)
#             # print('lat_data_for_icao_idx = ', lat_data_for_icao_idx)
#             # print('lat_data_for_icao_idx.shape = ', lat_data_for_icao_idx.shape)
#
#             # ax.plot(x_data_array, data[icao_idx], **plt_kwargs, label=str(track_id_array[icao_idx]))
#             # ax.scatter(x_data_array, data[icao_idx], **plt_kwargs, label=str(icao_array[icao_idx]))
#             # ax.plot(range(0, len(data[icao_idx])), data[icao_idx], alpha=0.2, **plt_kwargs, label=str(track_id_array[icao_idx]))
#
#             # used to print on plot the 'trajectory' legend only for the last trajectory instead of for all
#             if icao_idx != cluster_icao_idx[-1]:
#                 # print('trigger 1')
#                 # ax.plot(range(0, len(alt_data_for_icao_idx)), alt_data_for_icao_idx, c='gray', alpha=0.3, **plt_kwargs)
#                 # axs[0, 0].plot(range(0, len(alt_data_for_icao_idx)), alt_data_for_icao_idx, c='gray', alpha=0.3, **plt_kwargs)
#                 # axs[0, 1].plot(range(0, len(alt_data_for_icao_idx)), lon_data_for_icao_idx, c='gray', alpha=0.3, **plt_kwargs)
#                 # axs[0, 2].plot(range(0, len(alt_data_for_icao_idx)), lat_data_for_icao_idx, c='gray', alpha=0.3, **plt_kwargs)
#                 # ax1.plot(range(0, len(alt_data_for_icao_idx)), alt_data_for_icao_idx, c='gray', alpha=opacity, **plt_kwargs)
#                 # ax2.plot(range(0, len(alt_data_for_icao_idx)), lon_data_for_icao_idx, c='gray', alpha=opacity, **plt_kwargs)
#                 # ax3.plot(range(0, len(alt_data_for_icao_idx)), lat_data_for_icao_idx, c='gray', alpha=opacity, **plt_kwargs)
#                 opacity = 0.4
#                 linewidth = 1
#                 zorder = 1
#                 if features == ['lon', 'lat', 'alt']:
#                     ax1.plot(range(0, len(lon_data_for_icao_idx)), lon_data_for_icao_idx, c=color_list[cluster], linewidth=linewidth,alpha=opacity, zorder=zorder, **plt_kwargs)
#                     ax2.plot(range(0, len(lat_data_for_icao_idx)), lat_data_for_icao_idx, c=color_list[cluster], linewidth=linewidth,alpha=opacity, zorder=zorder, **plt_kwargs)
#                     ax3.plot(range(0, len(alt_data_for_icao_idx)), alt_data_for_icao_idx, c=color_list[cluster], linewidth=linewidth,alpha=opacity, zorder=zorder, **plt_kwargs)
#                 if features == ['lon', 'lat', 'alt', 'hdg']:
#                     ax1.plot(range(0, len(lon_data_for_icao_idx)), lon_data_for_icao_idx, c=color_list[cluster], linewidth=linewidth, alpha=opacity, zorder=zorder, **plt_kwargs)
#                     ax2.plot(range(0, len(lat_data_for_icao_idx)), lat_data_for_icao_idx, c=color_list[cluster], linewidth=linewidth, alpha=opacity, zorder=zorder, **plt_kwargs)
#                     ax3.plot(range(0, len(alt_data_for_icao_idx)), alt_data_for_icao_idx, c=color_list[cluster], linewidth=linewidth, alpha=opacity, zorder=zorder, **plt_kwargs)
#                     ax4.plot(range(0, len(hdg_data_for_icao_idx)), hdg_data_for_icao_idx, c=color_list[cluster], linewidth=linewidth, alpha=opacity, zorder=zorder, **plt_kwargs)
#             # else:
#             #     # print('trigger 2')
#             #     # ax.plot(range(0, len(alt_data_for_icao_idx)), alt_data_for_icao_idx, c='gray', alpha=0.3, label='trajectories', **plt_kwargs)
#             #     # axs[0, 0].plot(range(0, len(alt_data_for_icao_idx)), alt_data_for_icao_idx, c='gray', alpha=0.3, label='trajectories', **plt_kwargs)
#             #     # axs[0, 1].plot(range(0, len(alt_data_for_icao_idx)), lon_data_for_icao_idx, c='gray', alpha=0.3, label='trajectories', **plt_kwargs)
#             #     # axs[0, 2].plot(range(0, len(alt_data_for_icao_idx)), lat_data_for_icao_idx, c='gray', alpha=0.3, label='trajectories', **plt_kwargs)
#             #     # ax1.plot(range(0, len(alt_data_for_icao_idx)), alt_data_for_icao_idx, c='gray', alpha=opacity, label='cluster_'+str(i), **plt_kwargs)
#             #     # ax2.plot(range(0, len(alt_data_for_icao_idx)), lon_data_for_icao_idx, c='gray', alpha=opacity, label='cluster_'+str(i), **plt_kwargs)
#             #     # ax3.plot(range(0, len(alt_data_for_icao_idx)), lat_data_for_icao_idx, c='gray', alpha=opacity, label='cluster_'+str(i), **plt_kwargs)
#             #     ax1.plot(range(0, len(alt_data_for_icao_idx)), alt_data_for_icao_idx, c=color_list[cluster], alpha=opacity, label='cluster_'+str(i), **plt_kwargs)
#             #     ax2.plot(range(0, len(alt_data_for_icao_idx)), lon_data_for_icao_idx, c=color_list[cluster], alpha=opacity, label='cluster_'+str(i), **plt_kwargs)
#             #     ax3.plot(range(0, len(alt_data_for_icao_idx)), lat_data_for_icao_idx, c=color_list[cluster], alpha=opacity, label='cluster_'+str(i), **plt_kwargs)
#         # ax.plot(range(0, len(alt_data_for_icao_idx)), alt_centroids, c='red', label='centroid', **plt_kwargs)
#         # ax1.plot(range(0, len(alt_data_for_icao_idx)), alt_centroids, c='red', linewidth=2,label='centroid', **plt_kwargs)
#         # ax2.plot(range(0, len(lon_data_for_icao_idx)), lon_centroids, c='red', linewidth=2, label='centroid', **plt_kwargs)
#         # ax3.plot(range(0, len(lat_data_for_icao_idx)), lat_centroids, c='red', linewidth=2, label='centroid', **plt_kwargs)
#         linewidth = 4
#         zorder = 10
#         edgecolor = 'k'
#         if features == ['lon', 'lat', 'alt']:
#             ax1.plot(range(0, len(lon_data_for_icao_idx)), lon_centroids, c=color_list[cluster], linewidth=linewidth, path_effects=[pe.Stroke(linewidth=linewidth+2, foreground=edgecolor), pe.Normal()], zorder=zorder,label='Centroid of Cluster ' + str(i), **plt_kwargs)
#             ax2.plot(range(0, len(lat_data_for_icao_idx)), lat_centroids, c=color_list[cluster], linewidth=linewidth, path_effects=[pe.Stroke(linewidth=linewidth+2, foreground=edgecolor), pe.Normal()], zorder=zorder,label='Centroid of Cluster ' + str(i), **plt_kwargs)
#             ax3.plot(range(0, len(alt_data_for_icao_idx)), alt_centroids, c=color_list[cluster], linewidth=linewidth, path_effects=[pe.Stroke(linewidth=linewidth+2, foreground=edgecolor), pe.Normal()], zorder=zorder,label='Centroid of Cluster ' + str(i), **plt_kwargs)
#         if features == ['lon', 'lat', 'alt', 'hdg']:
#             ax1.plot(range(0, len(lon_data_for_icao_idx)), lon_centroids, c=color_list[cluster], linewidth=linewidth, path_effects=[pe.Stroke(linewidth=linewidth+2, foreground=edgecolor), pe.Normal()], zorder=zorder,label='Centroid of Cluster '+str(i), **plt_kwargs)
#             ax2.plot(range(0, len(lat_data_for_icao_idx)), lat_centroids, c=color_list[cluster], linewidth=linewidth, path_effects=[pe.Stroke(linewidth=linewidth+2, foreground=edgecolor), pe.Normal()], zorder=zorder, label='Centroid of Cluster '+str(i), **plt_kwargs)
#             ax3.plot(range(0, len(alt_data_for_icao_idx)), alt_centroids, c=color_list[cluster], linewidth=linewidth, path_effects=[pe.Stroke(linewidth=linewidth+2, foreground=edgecolor), pe.Normal()], zorder=zorder, label='Centroid of Cluster '+str(i), **plt_kwargs)
#             ax4.plot(range(0, len(hdg_data_for_icao_idx)), hdg_centroids, c=color_list[cluster], linewidth=linewidth, path_effects=[pe.Stroke(linewidth=linewidth+2, foreground=edgecolor), pe.Normal()], zorder=zorder, label='Centroid of Cluster '+str(i), **plt_kwargs)
#
#
#     # fig.suptitle('Cluster '+str(cluster_idx)+' out of '+str(cluster))
#     fig.suptitle('Number of Clusters = '+str(cluster_idx))
#     ax3.legend(loc='upper right').set_zorder(100)
#     # ax3.annotate(str(len(cluster_icao_idx)), xy=(1, 0), xycoords='axes fraction', fontsize=15, horizontalalignment='right', verticalalignment='bottom')
#
#     if features == ['lon', 'lat', 'alt']:
#         ax1.title.set_text('Longitude [deg]')
#         ax2.title.set_text('Latitude [deg]')
#         ax3.title.set_text('Altitude [m]')
#     if features == ['lon', 'lat', 'alt', 'hdg']:
#         ax1.title.set_text('Longitude')
#         ax2.title.set_text('Latitude')
#         ax3.title.set_text('Altitude')
#         ax4.title.set_text('Heading')
#
#     if features == ['lon', 'lat', 'alt']:
#         # ax1.set_ylabel('Normalized Values')
#         ax2.set_xlabel('Time Index')
#         # ax2.axes.get_yaxis().set_visible(False)
#         # ax3.axes.get_yaxis().set_visible(False)
#     if features == ['lon', 'lat', 'alt', 'hdg']:
#         ax1.set_ylabel('Normalized Values')
#         ax2.set_xlabel('Time Index')
#         # ax2.axes.get_yaxis().set_visible(False)
#         # ax3.axes.get_yaxis().set_visible(False)
#         # ax4.axes.get_yaxis().set_visible(False)
#     # fig.subplots_adjust(wspace=0, hspace=0)
#
#     fig_name = path + "nb clusters "+str(cluster_idx)+" features vs time.png"
#     fig.savefig(fig_name, dpi=200)
#     if features == ['lon', 'lat', 'alt']:
#         fig.clear(ax1)
#         fig.clear(ax2)
#         fig.clear(ax3)
#     if features == ['lon', 'lat', 'alt', 'hdg']:
#         fig.clear(ax1)
#         fig.clear(ax2)
#         fig.clear(ax3)
#         fig.clear(ax4)
#     # fig.clear(ax)
#     # fig.clear(axs)
#     # fig.clear(ax1)
#     # fig.clear(ax2)
#     # fig.clear(ax3)
#     plt.close(fig)
#     # ax = None
#
# for i, cluster in enumerate(np.unique(model_preds)):
#     multivariate_timeseries_plot4(multitimeseries_data=data_array, model_preds=model_preds, features=features_str_list,cluster_idx=cluster_id, centroids=cluster_centers, color_list=color_list, path=results_path)


# print('color_list = ', color_list)
# # color_list = plotly.colors.color_parser(color_list)
# color_list = plotly.colors.label_rgb(color_list)
# print('color_list = ', color_list)
#
# import plotly.graph_objects as go
# fig2 = go.Figure(data=[go.Scatter3d(
#     x=ac_info_df['Longitude [deg]'],
#     y=ac_info_df['Latitude [deg]'],
#     z=ac_info_df['Altitude [m]'],
#     mode='markers',
#     marker=dict(
#         size=2,
#         # color=ac_info_df['cluster'],                # set color to an array/list of desired values
#         color=plotly.colors.make_colorscale(ac_info_df['cluster']),                # set color to an array/list of desired values
#         # color=color_list,                # set color to an array/list of desired values
#         colorscale=color_list,   # choose a colorscale
#         # colorscale=ac_info_df['cluster'],   # choose a colorscale
#         opacity=0.7
#     )
# )])

print('ac_info_df cols =\n', list(ac_info_df.columns))

def select_individual_cluster_data(df, cluster_idx):
    selected_cluster_df = df[df['cluster'] == 'cluster_'+str(cluster_idx)]
    # print('selected_cluster_df =\n', selected_cluster_df)
    cols_to_remove = ['cluster', 'local time', 'time', 'icao24','TYPE AIRCRAFT', 'MODEL', 'datetime', 'track_id', 'track', 'CERTIFICATION','TYPE ENGINE','NO-ENG','NO-SEATS','AC-WEIGHT']
    data_col = selected_cluster_df.columns
    cols = list(data_col.values)
    cols = [x for x in cols if x in cols_to_remove]  # removes all cols except the ones we need
    data_col = data_col.drop(cols)
    selected_cluster_df = selected_cluster_df[data_col[:]].dropna()
    # print('selected_cluster_df of cluster ', str(cluster_idx),'=\n', selected_cluster_df)
    # selected_cluster_df.to_csv(path + 'cluster_' + str(cluster_idx) + 'df.csv')
    return selected_cluster_df

ac_info_df = ac_info_df.rename(columns={'lat':'Latitude [deg]',
                                  'lon':'Longitude [deg]',
                                  'geoaltitude':'Altitude [m]' ,
                                  'velocity':'Velocity [m/s]',
                                  'heading':'Heading [deg]',
                                  'turnrate':'Turn Rate [deg/s]',
                                  'acceleration':'Acceleration [m/s2]',
                                  'vertrate':'Vertical Rate [m/s]'})

def select_individual_cluster_data_info(df, cluster_idx):
    selected_cluster_df = df[df['cluster'] == 'cluster_'+str(cluster_idx)]
    # print('test =\n', selected_cluster_df)
    return selected_cluster_df

selected_cluster_df = select_individual_cluster_data(df=ac_info_df, cluster_idx=23)
selected_cluster_df = ac_info_df
# print('selected_cluster_df =\n', selected_cluster_df)
# ---------------------------------------------------------------------------------------------------------------------
# Histograms
# ---------------------------------------------------------------------------------------------------------------------

labels_list = list(selected_cluster_df.columns)
print('selected_cluster_df =\n', selected_cluster_df)
# print('labels_list =\n', labels_list)
# drop_cols_list =
# selected_cluster_df = selected_cluster_df.drop("Heading [deg]", axis='columns')
# selected_cluster_df = selected_cluster_df.drop(['Heading [deg]', 'Velocity [m/s]'], axis='columns')
# selected_cluster_df = selected_cluster_df.drop(['Latitude [deg]', 'Longitude [deg]', 'Altitude [m]', 'Heading [deg]', 'Velocity [m/s]'], axis='columns')
selected_cluster_df = selected_cluster_df[['Longitude [deg]', 'Latitude [deg]', 'Altitude [m]']]
print('selected_cluster_df =\n', selected_cluster_df)
col_str_list = list(selected_cluster_df.columns)
print('col_str_list = ', col_str_list)
nb_figs = len(col_str_list)
nb_cols, nb_rows = 2, int(nb_figs/2)
# nb_rows = 1 # used to force all figures to be on same row (use ony for small number of figs)
# nb_rows = nb_figs
while nb_rows*nb_cols < nb_figs:
    nb_rows = nb_rows+1

print('nb_rows = ', nb_rows)
print('nb_cols = ', nb_cols)
print('nb_figs = ', nb_figs)


bin_style = 250
font_size = 16
opacity = 0.7
# automatic version
# fig1_1, axes1_1 = plt.subplots(nrows=nb_rows, ncols=nb_cols)
# fig_counter = 0
# for row in np.arange(0, nb_rows):
#     for col in np.arange(0, nb_cols):
#         print('-------')
#         print('fig_counter =', fig_counter)
#         # boxplot_df[str(col_str_list[fig_counter])].plot.hist(bins=bin_style, ax=axes1_1[int(row), int(col)], title=str(col_str_list[fig_counter]))
#         var_str = str(col_str_list[fig_counter])
#         print('var_str     =', var_str)
#         # print('boxplot_df')
#         selected_cluster_df[var_str].plot.hist(bins=bin_style, ax=axes1_1[int(row), int(col)])
#         # sns.kdeplot(data=boxplot_df, x=var_str, ax=axes1_1[int(row), int(col)])
#         axes1_1[int(row), int(col)].set_xlabel(str(labels_list[fig_counter]))
#         fig_counter += 1
# manual version
# fig1_1, axes1_1 = plt.subplots(nrows=1, ncols=3)
# # selected_cluster_df['Longitude [deg]'].plot.hist(bins=bin_style, ax=axes1_1[1, 1])
# # selected_cluster_df['Latitude [deg]'].plot.hist(bins=bin_style, ax=axes1_1[1, 2])
# # selected_cluster_df['Altitude [m]'].plot.hist(bins=bin_style, ax=axes1_1[1, 3])
# selected_cluster_df['Longitude [deg]'].plot.hist(bins=bin_style, alpha=opacity, ax=axes1_1[0]) # need to use instead of above if only 1 row used
# selected_cluster_df['Latitude [deg]'].plot.hist(bins=bin_style, alpha=opacity, ax=axes1_1[1]) # need to use instead of above if only 1 row used
# selected_cluster_df['Altitude [m]'].plot.hist(bins=bin_style, alpha=opacity, ax=axes1_1[2]) # need to use instead of above if only 1 row used
# sns.kdeplot(data=selected_cluster_df, x='Longitude [deg]', ax=axes1_1[0])
# sns.kdeplot(data=selected_cluster_df, x='Latitude [deg]', ax=axes1_1[1])
# sns.kdeplot(data=selected_cluster_df, x='Altitude [m]', ax=axes1_1[2])
# # axes1_1[1, 1].set_xlabel('Longitude [deg]', fontsize=font_size)
# # axes1_1[1, 2].set_xlabel('Latitude [deg]', fontsize=font_size)
# # axes1_1[1, 3].set_xlabel('Altitude [m]', fontsize=font_size)
# axes1_1[0].set_xlabel('Longitude [deg]', fontsize=font_size)# need to use instead of above if only 1 row used
# axes1_1[1].set_xlabel('Latitude [deg]', fontsize=font_size)# need to use instead of above if only 1 row used
# axes1_1[2].set_xlabel('Altitude [m]', fontsize=font_size)# need to use instead of above if only 1 row used
# axes1_1[0].set_ylabel('Density', fontsize=font_size)# need to use instead of above if only 1 row used
# fig1_1.suptitle('Histograms')

# fig1_2, axes1_2 = plt.subplots(nrows=nb_rows, ncols=nb_cols)
# fig_counter = 0
# for row in np.arange(0, nb_rows):
#     for col in np.arange(0, nb_cols):
#         var_str = str(col_str_list[fig_counter])
#         # print('-------')
#         # print('scaled_boxplot_df')
#         # print('var_str =', var_str)
#         # scaled_boxplot_df[var_str].plot.hist(bins=bin_style,ax=axes1_2[int(row), int(col)], title=str(col_str_list[fig_counter]))
#         scaled_boxplot_df[var_str].plot.hist(bins=bin_style, ax=axes1_2[int(row), int(col)])
#         # axes1_2[int(row), int(col)].set_xlabel(str(col_str_list[fig_counter]))
#         # sns.kdeplot(data=scaled_boxplot_df, x=var_str, ax=axes1_2[int(row), int(col)])
#         axes1_2[int(row), int(col)].set_xlabel(str(labels_list[fig_counter]))
#         fig_counter += 1
# fig1_2.suptitle('Standardized Histograms')

# fig1_3, axes1_3 = plt.subplots(nrows=nb_rows, ncols=nb_cols)
# fig_counter = 0
# for row in np.arange(0, nb_rows):
#     for col in np.arange(0, nb_cols):
#         var_str = str(col_str_list[fig_counter])
#         # print( scipy.signal.find_peaks)
#         # sns.kdeplot(data=boxplot_df, x=var_str, ax=axes1_3[int(row), int(col)])
#         sns.kdeplot(data=selected_cluster_df, x=var_str, ax=axes1_3[int(row), int(col)])
#         # sns.kdeplot(data=scaled_boxplot_df, x=var_str, ax=axes1_3[int(row), int(col)])
#         axes1_3[int(row), int(col)].set_xlabel(str(labels_list[fig_counter]))
#         fig_counter += 1
# fig1_3.suptitle('KDE Plots')

# fig1_4, axes1_4 = plt.subplots(nrows=nb_rows, ncols=nb_cols)
# fig_counter = 0
# for row in np.arange(0, nb_rows):
#     for col in np.arange(0, nb_cols):
#         print('------')
#         print('fig_counter =', fig_counter)
#         try:
#             var_str = str(col_str_list[fig_counter])
#             print('var_str     =', var_str)
#             if nb_rows == 1:
#                 sns.distplot(selected_cluster_df[var_str], kde=True, bins=bin_style, ax=axes1_4[int(col)])
#             else:
#                 # print( scipy.signal.find_peaks)
#                 # sns.kdeplot(data=boxplot_df, x=var_str, ax=axes1_3[int(row), int(col)])
#                 # sns.kdeplot(data=scaled_boxplot_df, x=var_str, ax=axes1_3[int(row), int(col)])
#                 # sns.kdeplot(data=boxplot_df, x=var_str, ax=axes1_4[int(row), int(col)])
#                 sns.distplot(selected_cluster_df[var_str], kde=True, bins=bin_style, ax=axes1_4[int(row), int(col)])
#                 # axes1_4[int(row), int(col)].set_xlabel(str(labels_list[fig_counter]))
#         except IndexError:
#             print('! index error')
#             break
#         fig_counter += 1
# fig1_4, axes1_4 = plt.subplots(nrows=1, ncols=3)
# for col in range(0, nb_figs):
#     print('col = ', col)
#     # sns.histplot(selected_cluster_df['Longitude [deg]'], kde=True, bins=bin_style, ax=axes1_4[int(col)])
#     # sns.histplot(selected_cluster_df['Latitude [deg]'], kde=True, bins=bin_style, ax=axes1_4[int(col)])
#     sns.histplot(selected_cluster_df, x='Longitude [deg]',kde=True, bins=bin_style, ax=axes1_4[0, int(col)])
#     sns.histplot(selected_cluster_df, x='Latitude [deg]',kde=True, bins=bin_style, ax=axes1_4[0, int(col)])
#     sns.histplot(selected_cluster_df, x='Altitude [m]',kde=True, bins=bin_style, ax=axes1_4[0, int(col)])
# fig1_4.suptitle('Combined Histogram and KDE Plots')

# latest good one
fig1_4, axes1_4 = plt.subplots(nrows=1, ncols=3)
# sns.set(font_scale=2)
sns.histplot(selected_cluster_df, x='Longitude [deg]',kde=True, bins=bin_style, ax=axes1_4[0])
sns.histplot(selected_cluster_df, x='Latitude [deg]',kde=True, bins=bin_style, ax=axes1_4[1])
sns.histplot(selected_cluster_df, x='Altitude [m]',kde=True, bins=bin_style, ax=axes1_4[2])
axes1_4[0].set_xlabel('Longitude [deg]', fontsize=font_size)
axes1_4[1].set_xlabel('Latitude [deg]', fontsize=font_size)
axes1_4[2].set_xlabel('Altitude [m]', fontsize=font_size)
axes1_4[0].tick_params(axis='both', which='major', labelsize=font_size)
axes1_4[1].tick_params(axis='both', which='major', labelsize=font_size)
axes1_4[2].tick_params(axis='both', which='major', labelsize=font_size)
axes1_4[0].set_ylabel('Density', fontsize=font_size)
axes1_4[1].set(ylabel=None)
axes1_4[2].set(ylabel=None)
fig1_4.suptitle('Combined Histogram and KDE Plots')

# fig1_1.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
# fig1_2.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
# fig1_3.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
# fig1_4.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)

plt.show()

def plot_histo(df, cluster_idx, path, ax=None, plt_kwargs={}):
    if ax is None:
        # fig = plt.gcf().set_size_inches(10, 5)
        fig1_4 = plt.gcf()
        ax1_4 = plt.gca()
    fig1_4, axes1_4 = plt.subplots(nrows=nb_rows, ncols=nb_cols)

    fig_counter = 0
    for row in np.arange(0, nb_rows):
        for col in np.arange(0, nb_cols):
            # print('fig_counter = ', fig_counter)
            try:
                var_str = str(col_str_list[fig_counter])
                # print('var_str = ', var_str)
                # print('var_str =', var_str)
                # print( scipy.signal.find_peaks)
                # sns.kdeplot(data=boxplot_df, x=var_str, ax=axes1_3[int(row), int(col)])
                # sns.kdeplot(data=scaled_boxplot_df, x=var_str, ax=axes1_3[int(row), int(col)])
                # sns.kdeplot(data=boxplot_df, x=var_str, ax=axes1_4[int(row), int(col)])
                sns.distplot(df[var_str], kde=True, bins=bin_style, ax=axes1_4[int(row), int(col)], **plt_kwargs)
                # axes1_4[int(row), int(col)].set_xlabel(str(labels_list[fig_counter]))
            except IndexError:
                continue
            fig_counter += 1
    # axes1_4.legend(loc='best')
    # fig.suptitle('Cluster '+str(cluster_idx))
    # axes1_4.set_xlabel('Time')
    # axes1_4.set_ylabel('Altitude [m]')
    # fig_name = path + "alt_cluster_" + str(cluster_idx) + "_of_" + str(n_clusters) + ".png"
    if fig_counter == 7:
        print('trigger')
        axes1_4[3, 1].set_visible(False)
    if fig_counter == 3:
        print('trigger')
        axes1_4[1, 1].set_visible(False)

    fig1_4.subplots_adjust(left=0.2, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.8)

    fig_name = path + "histograms_cluster_" + str(cluster_idx) + ".png"
    fig1_4.savefig(fig_name, dpi=200)
    # fig1_4.clear(axes1_4)
    fig1_4.clear()
    plt.close(fig1_4)

# ---------------------------------------------------------------------------------------------------------------------
# Boxplots
# ---------------------------------------------------------------------------------------------------------------------

# fig2_1 = selected_cluster_df.boxplot()
#
# fig2_3, axes2_3 = plt.subplots(nrows=nb_rows, ncols=nb_cols)
# fig_counter = 0
# for row in np.arange(0, nb_rows):
#     for col in np.arange(0, nb_cols):
#         # data[str(predictors_str_list[fig_counter])].plot.hist(bins=50,ax=axes[int(row), int(col)], title=str(predictors_str_list[fig_counter]))
#         selected_cluster_df[str(col_str_list[fig_counter])].plot.box(grid=True, ax=axes2_3[int(row), int(col)])
#         # axes2_3[int(row), int(col)].set_xlabel(str(labels_list[fig_counter]))
#         fig_counter += 1

# fig2_4, axes2_4 = plt.subplots(nrows=nb_rows, ncols=nb_cols)
# fig_counter = 0
# for row in np.arange(0, nb_rows):
#     for col in np.arange(0, nb_cols):
#         # data[str(predictors_str_list[fig_counter])].plot.hist(bins=50,ax=axes[int(row), int(col)], title=str(predictors_str_list[fig_counter]))
#         scaled_boxplot_df[str(col_str_list[fig_counter])].plot.box(grid=True, ax=axes2_4[int(row), int(col)])
#         # axes2_4[int(row), int(col)].set_xlabel(str(labels_list[fig_counter]))
#         fig_counter += 1
#
# fig2_5 = scaled_boxplot_df.plot.box(grid=True)


fig2_6, axes2_6 = plt.subplots(nrows=1, ncols=3)
# sns.set(font_scale=2)
sns.boxplot(y=selected_cluster_df['Longitude [deg]'], ax=axes2_6[0])
sns.boxplot(y=selected_cluster_df['Latitude [deg]'], ax=axes2_6[1])
sns.boxplot(y=selected_cluster_df['Altitude [m]'], ax=axes2_6[2])
axes2_6[0].set_xlabel('Longitude [deg]', fontsize=font_size)
axes2_6[1].set_xlabel('Latitude [deg]', fontsize=font_size)
axes2_6[2].set_xlabel('Altitude [m]', fontsize=font_size)
axes2_6[0].tick_params(axis='both', which='major', labelsize=font_size)
axes2_6[1].tick_params(axis='both', which='major', labelsize=font_size)
axes2_6[2].tick_params(axis='both', which='major', labelsize=font_size)
axes2_6[0].set_ylabel('Density', fontsize=font_size)
axes2_6[1].set(ylabel=None)
axes2_6[2].set(ylabel=None)
axes2_6[0].xaxis.grid(True) # Show the vertical gridlines
axes2_6[1].xaxis.grid(True) # Show the vertical gridlines
axes2_6[2].xaxis.grid(True) # Show the vertical gridlines
axes2_6[0].yaxis.grid(True) # Hide the horizontal gridlines
axes2_6[1].yaxis.grid(True) # Hide the horizontal gridlines
axes2_6[2].yaxis.grid(True) # Hide the horizontal gridlines
# plt.show()

def plot_boxplot(df, cluster_idx, path, ax=None, plt_kwargs={}):
    if ax is None:
        # fig = plt.gcf().set_size_inches(10, 5)
        fig = plt.gcf()
        ax = plt.gca()
    fig2_1 = df.boxplot()
    fig_name = path + "boxplots_cluster_" + str(cluster_idx) + ".png"
    fig.savefig(fig_name, dpi=200)
    # fig1_4.clear(axes1_4)
    plt.close(fig2_1)

# ---------------------------------------------------------------------------------------------------------------------
# Correlation matrix
# ---------------------------------------------------------------------------------------------------------------------

# height_style = 1 # can also add height=1 to add separation between subplots
# # sns.set_style('whitegrid')
# sns.set_style('darkgrid')
# corner_style = True  # Set corner=True to plot only the lower triangle
# alpha_style = 0.1  # set transparency

# fig3_4 = sns.pairplot(temp_adsb_info_df, hue='CERTIFICATION', height=1, corner=corner_style, plot_kws={'s':1, 'linewidth': 0, 'alpha':alpha_style})
# fig3_4 = sns.pairplot(temp_adsb_info_df, hue='TYPE AIRCRAFT', height=1, corner=corner_style, plot_kws={'s':1, 'linewidth': 0, 'alpha': alpha_style})
# fig3_4 = sns.pairplot(temp_adsb_info_df, hue='MODEL', height=1, corner=corner_style, plot_kws={'s':1, 'linewidth': 0, 'alpha': alpha_style})
# fig3_4 = sns.pairplot(selected_cluster_df, height=1, corner=corner_style, plot_kws={'s':1, 'linewidth': 0, 'alpha': 0.05})
#
# # fig3_5 = sns.pairplot(temp_adsb_info_df, hue='AC-WEIGHT', hue_order=['Up to 12,499', '12,500 - 19,999', '20,000 and over'], height=1, corner=corner_style, plot_kws={'s':1, 'linewidth': 0})
# fig3_5 = sns.pairplot(temp_adsb_info_df, hue='AC-WEIGHT', height=1, corner=corner_style, plot_kws={'s':1, 'linewidth': 0, 'alpha':alpha_style})
#
# fig3_6 = sns.pairplot(temp_adsb_info_df, hue='TYPE ENGINE', height=1, corner=corner_style, plot_kws={'s':1, 'linewidth': 0, 'alpha':alpha_style})

# temp_adsb_info_df = adsb_info_df[adsb_info_df.columns.drop(['time', 'lat', 'lon', 'NO-SEATS'])[:]]
# fig3_7 = sns.pairplot(temp_adsb_info_df, hue='NO-ENG', height=1, corner=corner_style, plot_kws={'s':1, 'linewidth': 0, 'alpha':alpha_style})
#
# temp_adsb_info_df = adsb_info_df[adsb_info_df.columns.drop(['time', 'lat', 'lon', 'NO-ENG'])[:]]
# fig3_8 = sns.pairplot(temp_adsb_info_df, hue='NO-SEATS', height=1, corner=corner_style, plot_kws={'s':1, 'linewidth': 0, 'alpha':alpha_style})

# fig3_2 = sns.pairplot(adsb_df, corner=corner_style, height=height_style, plot_kws={'s':1, 'linewidth': 0})
# fig3_2.map_lower(sns.kdeplot, levels=5, color=".2")

# fig3_3 = sns.pairplot(adsb_df, corner=corner_style, diag_kind="kde",
#     diag_kws={"linewidth": 0, "shade": False})

# fig3_5 = sns.PairGrid(adsb_df, diag_sharey=False, height=height_style)
# fig3_5.map_upper(sns.scatterplot, s=1)
# fig3_5.map_lower(sns.kdeplot, levels=4)
# fig3_5.map_diag(sns.kdeplot, linewidth=1)

# fig3_6 = sns.pairplot(scaled_boxplot_df, kind="kde", corner=corner_style, height=height_style)


# # plt.matshow(corr)
# # corr.style.background_gradient(cmap='coolwarm')
# corr_matrix = selected_cluster_df.corr()
# corr_matrix_upper = np.triu(corr_matrix)
#
# # sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
# sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', mask=corr_matrix_upper)
#
#
# plt.show()

def plot_corr_matrix(df, cluster_idx, path, ax=None, plt_kwargs={}):
    if ax is None:
        # fig = plt.gcf().set_size_inches(10, 5)
        fig4_1 = plt.gcf()
        ax4_1 = plt.gca()
    corr_matrix = df.corr()
    corr_matrix_upper = np.triu(corr_matrix)
    # sns.heatmap(corr_matrix, annot=True, cmap='seismic', mask=corr_matrix_upper, **plt_kwargs)
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', mask=corr_matrix_upper, **plt_kwargs)
    # sns.heatmap(corr_matrix, annot=True, cmap='Blues', mask=corr_matrix_upper, **plt_kwargs)

    fig4_1.subplots_adjust(left=0.3, bottom=0.35, right=0.9, top=0.9, wspace=0.4, hspace=0.4)

    fig_name = path + "corr_matrix_cluster_" + str(cluster_idx) + ".png"
    fig4_1.savefig(fig_name, dpi=200)
    # fig1_4.clear(axes1_4)
    plt.close(fig4_1)

def plot_snscorr_matrix(df, cluster_idx, path, ax=None, plt_kwargs={}):
    if ax is None:
        # fig = plt.gcf().set_size_inches(10, 5)
        fig4_2 = plt.gcf()
        ax4_2 = plt.gca()

    # corr_matrix = df.corr()
    # corr_matrix_upper = np.triu(corr_matrix)
    # sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', mask=corr_matrix_upper, **plt_kwargs)

    # sns.pairplot(df, height=1, corner=True, plot_kws={'s':1, 'linewidth': 0, 'alpha': 0.05})
    # sns.pairplot(df, height=1, corner=True, **plt_kwargs)
    sns.pairplot(df, height=1, corner=True, plot_kws={'s':1, 'linewidth': 0, 'alpha': 0.05}, **plt_kwargs)

    fig_name = path + "snscorr_matrix_cluster_" + str(cluster_idx) + ".png"
    fig4_2.savefig(fig_name, dpi=200)
    # fig1_4.clear(axes1_4)
    plt.close(fig4_2)

def plot_alt_vs_time_per_cluster(df, cluster_idx, path, ax=None, plt_kwargs={}):
    if ax is None:
        # fig = plt.gcf().set_size_inches(10, 5)
        fig5_1 = plt.gcf()
        ax5_1 = plt.gca()

    for track_id in list(df.track_id.unique()):
        temp_df = df[df['track_id'] == track_id]
        # ax5_1.plot(temp_df.datetime, temp_df.geoaltitude, **plt_kwargs)
        ax5_1.plot(temp_df.index, temp_df.geoaltitude, **plt_kwargs)
    ax5_1.legend(loc='best')
    fig5_1.suptitle('Cluster '+str(cluster_idx))
    ax5_1.set_xlabel('Time')
    ax5_1.set_ylabel('Altitude [m]')

    # ax5_1.xaxis_date()
    # ax5_1.set_xticks(df['datetime'])
    # # ax5_1.xaxis.set_major_formatter(DateFormatter("%M:%D"))
    # # ax5_1.xaxis.set_minor_formatter(DateFormatter("%H"))
    # # # ax5_1.xaxis.set_minor_formatter(DateFormatter("%H:%M:%S"))
    # ax5_1.xaxis.set_major_formatter(DateFormatter("%H:%M:%S"))
    # _ = plt.xticks(rotation=45)

    fig_name = path + "alt_vs_time_cluster_" + str(cluster_idx) + ".png"
    fig5_1.savefig(fig_name, dpi=200)
    # fig1_4.clear(axes1_4)
    plt.close(fig5_1)

def plot_all_vs_time_per_cluster(df, cluster_idx, path, ax=None, plt_kwargs={}):
    if ax is None:
        # fig = plt.gcf().set_size_inches(10, 5)
        fig5_2 = plt.gcf()
        ax5_2 = plt.gca()

    fig5_2, axes5_2 = plt.subplots(nrows=nb_rows, ncols=nb_cols)

    # print('initial df =\n', df)
    # print(list(df.track_id.unique()))
    nb_unique_tracks = df.track_id.nunique()
    # print(nb_unique_tracks)

    # print('nb_rows = ', nb_rows)
    # print('nb_cols = ', nb_cols)

    fig_counter = 0
    for row in np.arange(0, nb_rows):
        for col in np.arange(0, nb_cols):
            # print('fig_counter = ', fig_counter, '/', len(col_str_list))
            if fig_counter <= len(col_str_list)-1:
                var_str = str(col_str_list[fig_counter])
                # print('var_str = ', var_str)
                for track_id in list(df.track_id.unique()):
                    # print('row = ', row)
                    # print('col = ', col)
                    temp_df = df[df['track_id'] == track_id]
                    # print('------------------------')
                    # print('row      = ', row)
                    # print('col      = ', col)
                    # print('track_id = ', track_id)
                    # print('var_str  = ', var_str)
                    # print('temp_df of ', str(track_id),'\n', temp_df)
                    # print('temp_df of ', str(var_str),'\n', temp_df[var_str])
                    # ax5_1.plot(temp_df.datetime, temp_df.geoaltitude, **plt_kwargs)
                    # axes5_2[row, col].plot(temp_df.index, temp_df[var_str], **plt_kwargs)
                    # axes5_2[row, col].plot(range(0, len(temp_df[var_str])), temp_df[var_str], linewidth=1, label=str(track_id), **plt_kwargs)
                    axes5_2[row, col].plot(range(0, len(temp_df[var_str])), temp_df[var_str], linewidth=1, label=str(track_id), **plt_kwargs)
                    # sns.distplot(df[var_str], kde=True, bins=bin_style, ax=axes5_2[int(row), int(col)], **plt_kwargs)
                    axes5_2[int(row), int(col)].set_ylabel(str(labels_list[fig_counter]), fontsize=7.5, **plt_kwargs)
                fig_counter += 1
    # print('final fig_counter =', fig_counter)
    if fig_counter == 7:
        print('trigger')
        axes5_2[3, 1].set_visible(False)
    if fig_counter == 3:
        print('trigger')
        axes5_2[1, 1].set_visible(False)
    # axes5_2[1,0].legend(loc='best')
    fig5_2.suptitle('Cluster '+str(cluster_idx)+' ('+str(nb_unique_tracks)+' trajectories)')
    # print('nb_cols= ', nb_cols)
    # print('col =', col)
    axes5_2[col, 0].set_xlabel('Time Index')
    axes5_2[col, 1].set_xlabel('Time Index')
    # axes5_2[].set_ylabel('Altitude [m]')

    # ax5_1.xaxis_date()
    # ax5_1.set_xticks(df['datetime'])
    # # ax5_1.xaxis.set_major_formatter(DateFormatter("%M:%D"))
    # # ax5_1.xaxis.set_minor_formatter(DateFormatter("%H"))
    # # # ax5_1.xaxis.set_minor_formatter(DateFormatter("%H:%M:%S"))
    # ax5_1.xaxis.set_major_formatter(DateFormatter("%H:%M:%S"))
    # _ = plt.xticks(rotation=45)

    fig5_2.subplots_adjust(left=0.2, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.8)

    fig_name = path + "all_vs_time_cluster_" + str(cluster_idx) + ".png"
    fig5_2.savefig(fig_name, dpi=200)
    # fig5_2.clear(axes5_2)
    plt.close(fig5_2)

from matplotlib.pyplot import cm

def plot_hdg_vs_time_all_clusters(df, path, ax=None, plt_kwargs={}):

    n_clusters = len(pd.unique(df['cluster']))
    # print('n_clusters = ', n_clusters)

    color_list = cm.Set3(np.linspace(0, 1, n_clusters))
    # print('color_list = ', color_list)
    # print('len color_list = ', len(color_list))

    # fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig, (ax1) = plt.subplots(1, 1)
    opacity = 0.7
    l_width = 1
    # for cluster_id in list(df.cluster.unique()):
    for i, cluster_id in enumerate(list(df.cluster.unique())):
        # print('cluster_id = ', cluster_id)
        # print('i          = ', i)
        for track_id in list(df.track_id.unique()):
            # print('track_id = ', track_id)
            temp_df = df[(df['track_id'] == track_id) & (df['cluster'] == cluster_id)]
            temp_df = df[df.track_id.str.startswith(track_id).values & df.cluster.str.startswith(cluster_id).values]
            if not temp_df.empty:
                # print('temp_df =\n', temp_df)
                ax1.plot(range(0, len(temp_df)), temp_df['Heading [deg]'], c=color_list[i], alpha=opacity, linewidth=l_width, **plt_kwargs)
                # ax2.plot(range(0, len(alt_data_for_icao_idx)), lon_data_for_icao_idx, c=color_list[cluster], alpha=opacity, **plt_kwargs)
                # ax3.plot(range(0, len(alt_data_for_icao_idx)), lat_data_for_icao_idx, c=color_list[cluster], alpha=opacity, **plt_kwargs)
            # else:
            #     print('temp_df is empty')

    ax1.title.set_text('Heading')
    ax1.set_xlabel('Time Index')
    ax1.set_ylabel('Heading [deg]')

    fig_name = path + "hdg vs time.png"
    fig.savefig(fig_name, dpi=200)

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

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(-1,1))

def plot_hdg_vs_time_all_clusters2(features, multitimeseries_data, ac_info_data, model_preds, centroids, cluster_idx, colors, path, ax=None, plt_kwargs={}):
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

def plot_alt_vs_time_all_clusters(df, path, ax=None, plt_kwargs={}):

    n_clusters = len(pd.unique(df['cluster']))
    # print('n_clusters = ', n_clusters)

    color_list = cm.Set3(np.linspace(0, 1, n_clusters))
    # print('color_list = ', color_list)
    # print('len color_list = ', len(color_list))

    # fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig, (ax1) = plt.subplots(1, 1)
    opacity = 0.7
    l_width = 1
    # for cluster_id in list(df.cluster.unique()):
    for i, cluster_id in enumerate(list(df.cluster.unique())):
        # print('cluster_id = ', cluster_id)
        # print('i          = ', i)
        for track_id in list(df.track_id.unique()):
            # print('track_id = ', track_id)
            temp_df = df[(df['track_id'] == track_id) & (df['cluster'] == cluster_id)]
            temp_df = df[df.track_id.str.startswith(track_id).values & df.cluster.str.startswith(cluster_id).values]
            if not temp_df.empty:
                # print('temp_df =\n', temp_df)
                ax1.plot(range(0, len(temp_df)), temp_df['Altitude [m]'], c=color_list[i], alpha=opacity, linewidth=l_width, **plt_kwargs)
                # ax2.plot(range(0, len(alt_data_for_icao_idx)), lon_data_for_icao_idx, c=color_list[cluster], alpha=opacity, **plt_kwargs)
                # ax3.plot(range(0, len(alt_data_for_icao_idx)), lat_data_for_icao_idx, c=color_list[cluster], alpha=opacity, **plt_kwargs)
            # else:
            #     print('temp_df is empty')

    ax1.title.set_text('Altitude')
    ax1.set_xlabel('Time Index')
    ax1.set_ylabel('Altitude [m]')

    fig_name = path + "alt vs time.png"
    fig.savefig(fig_name, dpi=200)

# print(list(cluster_data_df.columns))
# print(len(list(cluster_data_df.columns)))
# print(len(list(cluster_data_df.columns))-1)
# print(list(np.arange(0, len(list(cluster_data_df.columns)), 1)))
# cluster_idx_list = [7, 14, 16]
# cluster_idx_list = [24, 1, 26]
cluster_idx_list = list(np.arange(0, len(list(cluster_data_df.columns)), 1))
# results_path = r'C:/Users/nicol/Google Drive/PhD/Conferences & Papers/AIAA Aviation 2023/Code and results/2_2. Aircraft behaviour category detection/results visuals/'
results_path = r'C:/Users/nicol/Google Drive/PhD/Conferences & Papers/AIAA Journal/Code and results/results/visuals/'
# fig1.write_html(results_path+"2D_map_all_tracks.html")
# fig1.write_html(results_path+"nb clusters "+str(n_clusters)+"2D_map_all_tracks.html")
fig1.write_html(data_dir+"results/nb clusters "+str(n_clusters)+"_2D_map_all_tracks.html")
fig2.write_html(data_dir+"results/nb clusters "+str(n_clusters)+"_3D_map_all_tracks.html")

def plot_unique_counts_in_cluster(df, cluster_idx, path):

    # ['TYPE AIRCRAFT', 'CERTIFICATION', 'TYPE ENGINE', 'MODEL', 'NO-ENG', 'NO-SEATS', 'AC-WEIGHT']

    fig5_1 = px.bar(df['TYPE AIRCRAFT'].value_counts())
    fig5_2 = px.bar(df['CERTIFICATION'].value_counts())
    fig5_3 = px.bar(df['TYPE ENGINE'].value_counts())
    fig5_4 = px.bar(df['MODEL'].value_counts())
    fig5_5 = px.bar(df['NO-ENG'].value_counts())
    fig5_6 = px.bar(df['NO-SEATS'].value_counts())
    fig5_7 = px.bar(df['AC-WEIGHT'].value_counts())

    fig5_1.write_html(results_path +"cluster_"+str(cluster_idx)+"_unique_typeac_counts.html")
    fig5_2.write_html(results_path +"cluster_"+str(cluster_idx)+"_unique_cert_counts.html")
    fig5_3.write_html(results_path +"cluster_"+str(cluster_idx)+"_unique_engine_counts.html")
    fig5_4.write_html(results_path +"cluster_"+str(cluster_idx)+"_unique_model_counts.html")
    fig5_5.write_html(results_path +"cluster_"+str(cluster_idx)+"_unique_nbengine_counts.html")
    fig5_6.write_html(results_path +"cluster_"+str(cluster_idx)+"_unique_seats_counts.html")
    fig5_7.write_html(results_path +"cluster_"+str(cluster_idx)+"_unique_weight_counts.html")

def save_unique_counts_in_cluster_to_df(df, cluster_idx, path):

    # ['TYPE AIRCRAFT', 'CERTIFICATION', 'TYPE ENGINE', 'MODEL', 'NO-ENG', 'NO-SEATS', 'AC-WEIGHT']
    unique_counts_in_cluster_df = pd.DataFrame()
    # temp_df = pd.DataFrame({'TYPE AIRCRAFT': df['TYPE AIRCRAFT'].value_counts()})
    temp_df = pd.DataFrame({'TYPE AIRCRAFT': df.groupby('TYPE AIRCRAFT')['icao24'].nunique()})
    unique_counts_in_cluster_df = pd.concat([unique_counts_in_cluster_df, temp_df], axis=1)
    # temp_df = pd.DataFrame({'CERTIFICATION': df['CERTIFICATION'].value_counts()})
    temp_df = pd.DataFrame({'CERTIFICATION': df.groupby('CERTIFICATION')['icao24'].nunique()})
    unique_counts_in_cluster_df = pd.concat([unique_counts_in_cluster_df, temp_df], axis=1)
    # temp_df = pd.DataFrame({'TYPE ENGINE': df['TYPE ENGINE'].value_counts()})
    temp_df = pd.DataFrame({'TYPE ENGINE': df.groupby('TYPE ENGINE')['icao24'].nunique()})
    unique_counts_in_cluster_df = pd.concat([unique_counts_in_cluster_df, temp_df], axis=1)
    # temp_df = pd.DataFrame({'MODEL': df['MODEL'].value_counts()})
    temp_df = pd.DataFrame({'MODEL': df.groupby('MODEL')['icao24'].nunique()})
    unique_counts_in_cluster_df = pd.concat([unique_counts_in_cluster_df, temp_df], axis=1)
    # temp_df = pd.DataFrame({'NO-ENG': df['NO-ENG'].value_counts()})
    temp_df = pd.DataFrame({'NO-ENG': df.groupby('NO-ENG')['icao24'].nunique()})
    unique_counts_in_cluster_df = pd.concat([unique_counts_in_cluster_df, temp_df], axis=1)
    # temp_df = pd.DataFrame({'NO-SEATS': df['NO-SEATS'].value_counts()})
    temp_df = pd.DataFrame({'NO-SEATS': df.groupby('NO-SEATS')['icao24'].nunique()})
    unique_counts_in_cluster_df = pd.concat([unique_counts_in_cluster_df, temp_df], axis=1)
    # temp_df = pd.DataFrame({'AC-WEIGHT': df['AC-WEIGHT'].value_counts()})
    temp_df = pd.DataFrame({'AC-WEIGHT': df.groupby('AC-WEIGHT')['icao24'].nunique()})
    unique_counts_in_cluster_df = pd.concat([unique_counts_in_cluster_df, temp_df], axis=1)

    print('unique_counts_in_cluster_df =\n', unique_counts_in_cluster_df)
    unique_counts_in_cluster_df.to_csv(path + "cluster_"+str(cluster_idx)+"_unique_counts_in_cluster_df.csv")
    # unique_counts_in_cluster_df.to_csv(path + "cluster_"+str(cluster_idx)+"_unique_counts_in_cluster_df.csv")

# plots for all clusters combined
# plot_hdg_vs_time_all_clusters(df=ac_info_df, path=results_path)
# plot_alt_vs_time_all_clusters(df=ac_info_df, path=results_path)
# plot_hdg_vs_time_all_clusters2(multitimeseries_data=,
#                                ac_info_data=ac_info_data,
#                                model_preds=,
#                                centroids=,
#                                cluster_idx=,
#                                path=results_path)

# # indivual plots for each cluster
# for cluster_idx in cluster_idx_list:
#     print('cluster = ', cluster_idx)
#     selected_df = select_individual_cluster_data(df=ac_info_df, cluster_idx=cluster_idx)
#     selected_df_info = select_individual_cluster_data_info(df=ac_info_df, cluster_idx=cluster_idx)
#     print('selected_df = \n', selected_df)
#     print('selected_df_info = \n', selected_df_info)
    # plot_histo(df=selected_df, cluster_idx=cluster_idx, path=results_path)
    # plot_boxplot(df=select_individual_cluster_data(df=ac_info_df, cluster_idx=cluster_idx), cluster_idx=cluster_idx, path=results_path)
    # plot_corr_matrix(df=selected_df, cluster_idx=cluster_idx, path=results_path)
    # plot_snscorr_matrix(df=selected_df, cluster_idx=cluster_idx, path=results_path)
    # plot_unique_counts_in_cluster(df=selected_df_info, cluster_idx=cluster_idx, path=results_path)
    # save_unique_counts_in_cluster_to_df(df=selected_df_info, cluster_idx=cluster_idx, path=results_path)
    # plot_all_vs_time_per_cluster(df=selected_df_info, cluster_idx=cluster_idx, path=results_path)
    # plot_all_vs_time_per_cluster2(df=selected_df_info, cluster_idx=cluster_idx, path=results_path)
    # plot_alt_vs_time_per_cluster(df=selected_df_info, cluster_idx=cluster_idx, path=results_path)



