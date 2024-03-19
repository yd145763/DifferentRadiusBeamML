# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 10:03:12 2024

@author: limyu
"""

import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from sklearn.metrics import mean_absolute_error
from scipy import interpolate
from scipy.signal import chirp, find_peaks, peak_widths
from scipy.stats import linregress
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as ticker
import time
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
import seaborn as sns

file_list = "grating12pitch100", "grating12_11pitch6_4", "grating012umpitch05dutycycle60um"
file_path = "C:\\Users\\limyu\\Google Drive\\3D Data ML Beam\\DNNPredictBeamDataGPU_4TrainingRadius\\"
file_path2 = "C:\\Users\\limyu\\Downloads\\"
steps = 20
df_main = pd.DataFrame()

R = 20, 30, 40, 50, 60
R = pd.Series(R)
R_norm = (R - min(R))/(max(R) - min(R))
R_norm = round(R_norm, 5)
R_validation = 50
#R_prediction = 30
R_validation_norm = round((R_validation - min(R))/(max(R) - min(R)), 5)
#R_prediction_norm = round((R_prediction - min(R))/(max(R) - min(R)), 5)
R_training = R[R != R_validation]
#R_training = R_training[R_training != R_prediction]
R_training_norm = R_norm[R_norm != R_validation_norm]
#R_training_norm = R_training_norm[R_training_norm != R_prediction_norm]
R_training_norm = round(R_training_norm, 5)
Data_Source_Verification = False
Plot_from_saved_files = False
Plot_all_i = False
Plot_selected_i = True
Plot_ape = False

df_optimization = pd.read_csv(file_path+"optimization_result.csv")
df_optimization = df_optimization.iloc[:, 1:]

mat = df_optimization.pivot('nodes', 'layers', 'ape_list')
mat_list = mat.values.tolist()

plt.figure(figsize=(6, 4))
ax = sns.heatmap(mat_list, annot=True, cmap='twilight_shifted', fmt=".1f")

mat.columns
ax.set_xticklabels(mat.columns, fontweight="bold", fontsize = 12)
ax.set_yticklabels(mat.index, fontweight="bold", fontsize = 12)
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=12, width=2, length=6, pad=10, direction="out", colors="black", labelcolor="black")
for t in cbar.ax.get_yticklabels():
    t.set_fontweight("bold")
cbar.ax.set_title("Average\nPercentage\nError (%)", fontweight="bold")
font = {'color': 'black', 'weight': 'bold', 'size': 12}
ax.set_ylabel("Nodes (N)", fontdict=font)
ax.set_xlabel("Layers |(L)", fontdict=font)
#ax.tick_params(axis='both', labelsize=12, weight='bold')
for i, text in enumerate(ax.texts):
    text.set_fontsize(12)
for i, text in enumerate(ax.texts):
    text.set_fontweight('bold')
plt.show()
plt.close()

i = 310
for r in R:
    file = "grating012umpitch05dutycycle"+str(r)+"um"
    hdf5_file = file_path2+file+".h5"
    # Load the h5 file
    with h5py.File(hdf5_file, 'r') as f:
        # Get the dataset
        dset = f[file]
        # Load the dataset into a numpy array
        arr_3d_loaded = dset[()]
        
    df_df = arr_3d_loaded[:,:,i]
        
    x1 = np.linspace(0, 50, num=df_df.shape[1])
    y = np.linspace(-25, 25, num =df_df.shape[0])
    
    colorbarmax = df_df.max().max()
    
    X,Y = np.meshgrid(x1,y)
    fig = plt.figure(figsize=(5, 4))
    ax = plt.axes()
    cp=ax.contourf(X,Y,df_df, 200, zdir='z', offset=-100, cmap='twilight_shifted')
    clb=fig.colorbar(cp, ticks=(np.around(np.linspace(0.0, colorbarmax, num=6), decimals=3)).tolist())
    clb.ax.set_title('Electric Field (eV)', fontweight="bold")
    for l in clb.ax.yaxis.get_ticklabels():
        l.set_weight("bold")
        l.set_fontsize(15)
    ax.set_xlabel('x-position (µm)', fontsize=20, fontweight="bold", labelpad=1)
    ax.set_ylabel('y-position (µm)', fontsize=20, fontweight="bold", labelpad=1)
    ax.xaxis.label.set_fontsize(20)
    ax.xaxis.label.set_weight("bold")
    ax.yaxis.label.set_fontsize(20)
    ax.yaxis.label.set_weight("bold")
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_yticklabels(ax.get_yticks(), weight='bold')
    ax.set_xticklabels(ax.get_xticks(), weight='bold')
    ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    plt.show()
    plt.close()

import math
def calculate_rms_error(data):
    mean = sum(data) / len(data)
    sum_squared_diff = sum((value - mean) ** 2 for value in data)
    mean_squared_diff = sum_squared_diff / len(data)
    rms = math.sqrt(mean_squared_diff)
    return rms

z_cutoff = 100
I = np.arange(z_cutoff,arr_3d_loaded.shape[2],1)

if Data_Source_Verification ==True:
    
    df_main_main_full = pd.DataFrame()
    
    for r in R:
        file = "grating012umpitch05dutycycle"+str(r)+"um"
        hdf5_file = file_path2+file+".h5"
        # Load the h5 file
        with h5py.File(hdf5_file, 'r') as f:
            # Get the dataset
            dset = f[file]
            # Load the dataset into a numpy array
            arr_3d_loaded = dset[()]
        df_main_main = pd.DataFrame()
        for i in I[::steps]:  
            N = np.arange(0,df_df.shape[1],1)
            
            df_df = arr_3d_loaded[:,:,i]
            df_df = pd.DataFrame(df_df)
        
            df_main = pd.DataFrame() 
            for n in N[::steps]:
                x1 = np.linspace(0, 50, num=df_df.shape[1])[::steps]
                y = np.linspace(-25, 25, num =df_df.shape[0])[::steps]
                df = pd.DataFrame()
                df['y'] = y
                df['radius'] = r
                df['n'] = n
                df['i'] = i
                e = df_df[n]
                e = e[::steps]
                e.reset_index(drop=True, inplace=True)
                df['e'] = e
                df_main = pd.concat([df_main, df], axis= 0)
                print("{}-radius-{}-n-{}-i".format(r, n, i))
            df_main_main = pd.concat([df_main_main, df_main], axis= 0)
        df_main_main_full = pd.concat([df_main_main_full, df_main_main], axis= 0)
    
    for i in I[::steps]: 
        df_main_main = df_main_main_full[df_main_main_full['i'] == i]
    
            
        df_main_contour = pd.DataFrame()
        
        for n in N[::steps]:
            df_main = df_main_main[df_main_main['n'] == n]
            df = df_main[df_main['radius'] == 30]
        
            fig = plt.figure(figsize=(7, 4))
            ax = plt.axes()
            ax.plot(df['y'], df['e'], color = "blue")
            #graph formatting     
            ax.tick_params(which='major', width=2.00)
            ax.tick_params(which='minor', width=2.00)
            ax.xaxis.label.set_fontsize(15)
            ax.xaxis.label.set_weight("bold")
            ax.yaxis.label.set_fontsize(15)
            ax.yaxis.label.set_weight("bold")
            ax.tick_params(axis='both', which='major', labelsize=15)
            ax.set_yticklabels(ax.get_yticks(), weight='bold')
            ax.set_xticklabels(ax.get_xticks(), weight='bold')
            ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.3f}'))
            ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.spines['bottom'].set_linewidth(2)
            ax.spines['left'].set_linewidth(2)
            plt.xlabel("x-position (µm)")
            plt.ylabel("E-Field (eV)")
            plt.legend(["Actual", "Prediction"], prop={'weight': 'bold','size': 10}, loc = "upper left")
            plt.title("Actual/Prediction\n"+"n="+str(n)+"\n"+"i="+str(i), fontweight = 'bold')
            plt.show()
            plt.close()
            
            df_main_contour[n] = df['e']
        
        x1 = np.linspace(0, 50, num=df_main_contour.shape[1])
        y = np.linspace(-25, 25, num =df_main_contour.shape[0])
        
        colorbarmax = df_main_contour.max().max()
        
        X,Y = np.meshgrid(x1,y)
        fig = plt.figure(figsize=(5, 4))
        ax = plt.axes()
        cp=ax.contourf(X,Y,df_main_contour, 200, zdir='z', offset=-100, cmap='twilight_shifted')
        clb=fig.colorbar(cp, ticks=(np.around(np.linspace(0.0, colorbarmax, num=6), decimals=3)).tolist())
        clb.ax.set_title('Electric Field (eV)', fontweight="bold")
        for l in clb.ax.yaxis.get_ticklabels():
            l.set_weight("bold")
            l.set_fontsize(15)
        ax.set_xlabel('x-position (µm)', fontsize=20, fontweight="bold", labelpad=1)
        ax.set_ylabel('y-position (µm)', fontsize=20, fontweight="bold", labelpad=1)
        ax.xaxis.label.set_fontsize(20)
        ax.xaxis.label.set_weight("bold")
        ax.yaxis.label.set_fontsize(20)
        ax.yaxis.label.set_weight("bold")
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.set_yticklabels(ax.get_yticks(), weight='bold')
        ax.set_xticklabels(ax.get_xticks(), weight='bold')
        ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
        ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
        plt.show()
        plt.close()

#===================================

if Plot_from_saved_files == True:
    I = np.arange(100, 320, 20)
    for i in I:
        df_df_predicted = pd.read_csv(file_path+"3dpredictedmultibeam_model_nodes25_layers10_i"+str(i)+".csv")
        df_df_predicted = df_df_predicted.iloc[:, 1:]
        df_df_actual = pd.read_csv(file_path+"3dactualmultibeam_model_nodes25_layers10_i"+str(i)+".csv")
        df_df_actual = df_df_actual.iloc[:, 1:]
        N = pd.Series(df_df_actual.columns)
        y = np.linspace(-25, 25, num =df_df_actual.shape[0])
        x = np.linspace(0, 50, num=df_df_actual.shape[1])
        
        for n in N:
        
            e_actual = pd.Series(df_df_actual[str(n)])
            e_predicted = pd.Series(df_df_predicted[str(n)])
            
            fig = plt.figure(figsize=(7, 4))
            ax = plt.axes()
            ax.plot(y, e_actual, color = "blue")
            ax.plot(y, e_predicted, color = "red")
            #graph formatting     
            ax.tick_params(which='major', width=2.00)
            ax.tick_params(which='minor', width=2.00)
            ax.xaxis.label.set_fontsize(15)
            ax.xaxis.label.set_weight("bold")
            ax.yaxis.label.set_fontsize(15)
            ax.yaxis.label.set_weight("bold")
            ax.tick_params(axis='both', which='major', labelsize=15)
            ax.set_yticklabels(ax.get_yticks(), weight='bold')
            ax.set_xticklabels(ax.get_xticks(), weight='bold')
            ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.3f}'))
            ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.spines['bottom'].set_linewidth(2)
            ax.spines['left'].set_linewidth(2)
            plt.xlabel("x-position (µm)")
            plt.ylabel("E-Field (eV)")
            plt.legend(["Actual", "Prediction"], prop={'weight': 'bold','size': 10}, loc = "upper left")
            plt.title("Actual/Prediction\n"+"n="+str(n)+"\n"+"i="+str(i), fontweight = 'bold')
            plt.show()
            plt.close()
        
        colorbarmax = df_df_actual.max().max()
        
        X,Y = np.meshgrid(x,y)
        fig = plt.figure(figsize=(5, 4))
        ax = plt.axes()
        cp=ax.contourf(X,Y,df_df_actual, 200, zdir='z', offset=-100, cmap='twilight_shifted')
        clb=fig.colorbar(cp, ticks=(np.around(np.linspace(0.0, colorbarmax, num=6), decimals=3)).tolist())
        clb.ax.set_title('Electric Field (eV)', fontweight="bold")
        for l in clb.ax.yaxis.get_ticklabels():
            l.set_weight("bold")
            l.set_fontsize(15)
        ax.set_xlabel('x-position (µm)', fontsize=20, fontweight="bold", labelpad=1)
        ax.set_ylabel('y-position (µm)', fontsize=20, fontweight="bold", labelpad=1)
        ax.xaxis.label.set_fontsize(20)
        ax.xaxis.label.set_weight("bold")
        ax.yaxis.label.set_fontsize(20)
        ax.yaxis.label.set_weight("bold")
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.set_yticklabels(ax.get_yticks(), weight='bold')
        ax.set_xticklabels(ax.get_xticks(), weight='bold')
        ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
        ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
        plt.show()
        plt.close()
        
        colorbarmax = df_df_predicted.max().max()
        
        X,Y = np.meshgrid(x,y)
        fig = plt.figure(figsize=(5, 4))
        ax = plt.axes()
        cp=ax.contourf(X,Y,df_df_predicted, 200, zdir='z', offset=-100, cmap='twilight_shifted')
        clb=fig.colorbar(cp, ticks=(np.around(np.linspace(0.0, colorbarmax, num=6), decimals=3)).tolist())
        clb.ax.set_title('Electric Field (eV)', fontweight="bold")
        for l in clb.ax.yaxis.get_ticklabels():
            l.set_weight("bold")
            l.set_fontsize(15)
        ax.set_xlabel('x-position (µm)', fontsize=20, fontweight="bold", labelpad=1)
        ax.set_ylabel('y-position (µm)', fontsize=20, fontweight="bold", labelpad=1)
        ax.xaxis.label.set_fontsize(20)
        ax.xaxis.label.set_weight("bold")
        ax.yaxis.label.set_fontsize(20)
        ax.yaxis.label.set_weight("bold")
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.set_yticklabels(ax.get_yticks(), weight='bold')
        ax.set_xticklabels(ax.get_xticks(), weight='bold')
        ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
        ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
        plt.show()
        plt.close()
    
    """
#===================================
I = np.arange(100, 320, 10)
for i in I:
    df_df_predicted = pd.read_csv(file_path+"optimized3dpredictedmultibeam_model_nodes25_layers10_i"+str(i)+".csv")
    df_df_predicted = df_df_predicted.iloc[:, 1:]
    df_df_actual = pd.read_csv(file_path+"optimized3dactualmultibeam_model_nodes25_layers10_i"+str(i)+".csv")
    df_df_actual = df_df_actual.iloc[:, 1:]
    N = pd.Series(df_df_actual.columns)
    y = np.linspace(-25, 25, num =df_df_actual.shape[0])
    x = np.linspace(0, 50, num=df_df_actual.shape[1])
    
    for n in N:
    
        e_actual = pd.Series(df_df_actual[str(n)])
        e_predicted = pd.Series(df_df_predicted[str(n)])
        
        fig = plt.figure(figsize=(7, 4))
        ax = plt.axes()
        ax.plot(y, e_actual, color = "blue")
        ax.plot(y, e_predicted, color = "red")
        #graph formatting     
        ax.tick_params(which='major', width=2.00)
        ax.tick_params(which='minor', width=2.00)
        ax.xaxis.label.set_fontsize(15)
        ax.xaxis.label.set_weight("bold")
        ax.yaxis.label.set_fontsize(15)
        ax.yaxis.label.set_weight("bold")
        ax.tick_params(axis='both', which='major', labelsize=15)
        ax.set_yticklabels(ax.get_yticks(), weight='bold')
        ax.set_xticklabels(ax.get_xticks(), weight='bold')
        ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.3f}'))
        ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines['bottom'].set_linewidth(2)
        ax.spines['left'].set_linewidth(2)
        plt.xlabel("x-position (µm)")
        plt.ylabel("E-Field (eV)")
        plt.legend(["Actual", "Prediction"], prop={'weight': 'bold','size': 10}, loc = "upper left")
        plt.title("Actual/Prediction\n"+"n="+str(n)+"\n"+"i="+str(i), fontweight = 'bold')
        plt.show()
        plt.close()
    
    colorbarmax = df_df_actual.max().max()
    
    X,Y = np.meshgrid(x,y)
    fig = plt.figure(figsize=(5, 4))
    ax = plt.axes()
    cp=ax.contourf(X,Y,df_df_actual, 200, zdir='z', offset=-100, cmap='twilight_shifted')
    clb=fig.colorbar(cp, ticks=(np.around(np.linspace(0.0, colorbarmax, num=6), decimals=3)).tolist())
    clb.ax.set_title('Electric Field (eV)', fontweight="bold")
    for l in clb.ax.yaxis.get_ticklabels():
        l.set_weight("bold")
        l.set_fontsize(15)
    ax.set_xlabel('x-position (µm)', fontsize=20, fontweight="bold", labelpad=1)
    ax.set_ylabel('y-position (µm)', fontsize=20, fontweight="bold", labelpad=1)
    ax.xaxis.label.set_fontsize(20)
    ax.xaxis.label.set_weight("bold")
    ax.yaxis.label.set_fontsize(20)
    ax.yaxis.label.set_weight("bold")
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_yticklabels(ax.get_yticks(), weight='bold')
    ax.set_xticklabels(ax.get_xticks(), weight='bold')
    ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    plt.title("Actual Beam (i = "+str(i)+")\n", fontweight = 'bold', fontsize=15)
    plt.show()
    plt.close()
    
    time.sleep(2)
    
    colorbarmax = df_df_predicted.max().max()
    
    X,Y = np.meshgrid(x,y)
    fig = plt.figure(figsize=(5, 4))
    ax = plt.axes()
    cp=ax.contourf(X,Y,df_df_predicted, 200, zdir='z', offset=-100, cmap='twilight_shifted')
    clb=fig.colorbar(cp, ticks=(np.around(np.linspace(0.0, colorbarmax, num=6), decimals=3)).tolist())
    clb.ax.set_title('Electric Field (eV)', fontweight="bold")
    for l in clb.ax.yaxis.get_ticklabels():
        l.set_weight("bold")
        l.set_fontsize(15)
    ax.set_xlabel('x-position (µm)', fontsize=20, fontweight="bold", labelpad=1)
    ax.set_ylabel('y-position (µm)', fontsize=20, fontweight="bold", labelpad=1)
    ax.xaxis.label.set_fontsize(20)
    ax.xaxis.label.set_weight("bold")
    ax.yaxis.label.set_fontsize(20)
    ax.yaxis.label.set_weight("bold")
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_yticklabels(ax.get_yticks(), weight='bold')
    ax.set_xticklabels(ax.get_xticks(), weight='bold')
    ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    plt.title("Predicted Beam (i = "+str(i)+")\n", fontweight = 'bold', fontsize=15)
    plt.show()
    plt.close()
    
    time.sleep(2)
    
#====================================================

"""

if Plot_all_i == True:
    steps = 20
    import tensorflow as tf
    
    min_index = df_optimization['ape_list'].idxmin()
    layer_size_op = df_optimization.iloc[min_index, :]
    layer_size_op = int(layer_size_op['nodes']) 
    dense_layer_op = df_optimization.iloc[min_index, :]
    dense_layer_op = int(dense_layer_op['layers']) 
    
    I = np.arange(100, 301, 1)
    
    # Load the saved model
    model = tf.keras.models.load_model(file_path+"3dmultisinglebeam_model_nodes"+str(layer_size_op)+"_layers"+str(dense_layer_op))
    
    df_main_main_full = pd.DataFrame()
    
    for r in R:
        file = "grating012umpitch05dutycycle"+str(r)+"um"
        hdf5_file = file_path2+file+".h5"
        # Load the h5 file
        with h5py.File(hdf5_file, 'r') as f:
            # Get the dataset
            dset = f[file]
            # Load the dataset into a numpy array
            arr_3d_loaded = dset[()]
        df_main_main = pd.DataFrame()
        for i in I[::steps]:  
            N = np.arange(0,df_df.shape[1],1)
            
            df_df = arr_3d_loaded[:,:,i]
            df_df = pd.DataFrame(df_df)
        
            df_main = pd.DataFrame() 
            for n in N[::steps]:
                x1 = np.linspace(0, 50, num=df_df.shape[1])[::steps]
                y = np.linspace(-25, 25, num =df_df.shape[0])[::steps]
                df = pd.DataFrame()
                df['y'] = y
                df['radius'] = r
                df['n'] = n
                df['i'] = i
                e = df_df[n]
                e = e[::steps]
                e.reset_index(drop=True, inplace=True)
                df['e'] = e
                df_main = pd.concat([df_main, df], axis= 0)
                print("{}-radius-{}-n-{}-i".format(r, n, i))
            df_main_main = pd.concat([df_main_main, df_main], axis= 0)
        df_main_main_full = pd.concat([df_main_main_full, df_main_main], axis= 0)
    
            
    scaler = MinMaxScaler()
    df_main_main_full_norm = scaler.fit_transform(df_main_main_full)
    df_main_main_full_norm = pd.DataFrame(df_main_main_full_norm)
    df_main_main_full_norm = df_main_main_full_norm.round(5)
    
    df_training_norm = pd.DataFrame()
    for r in R_training_norm:
        df_intermediate = df_main_main_full_norm[df_main_main_full_norm[1] == r]
        df_training_norm = pd.concat([df_training_norm, df_intermediate], axis= 0)
    
    df_validation_norm = df_main_main_full_norm[df_main_main_full_norm[1] == R_validation_norm]
    
    
    X_train = df_training_norm.iloc[:, 0:-1]
    X_train.reset_index(drop=True, inplace=True)
    #X_train = X_train.values.astype(float)
    
    y_train = df_training_norm.iloc[:, -1]
    y_train = pd.DataFrame(y_train)
    y_train.reset_index(drop=True, inplace=True)
    
    X_test = df_validation_norm.iloc[:, 0:-1]
    X_test.reset_index(drop=True, inplace=True)
    #X_test = X_test.values.astype(float)
    
    y_test = df_validation_norm.iloc[:, -1]
    y_test = pd.DataFrame(y_test)
    y_test.reset_index(drop=True, inplace=True)
    
    predictions = model.predict(X_test)
    
    X_test.reset_index(drop=True, inplace=True)
    predictions = pd.DataFrame(predictions)
    predictions.reset_index(drop=True, inplace=True)
    df_predicted_norm = pd.concat([X_test, predictions], axis= 1)
    
    df_predicted = scaler.inverse_transform(df_predicted_norm)
    df_predicted[:, 2] = np.round(df_predicted[:, 2], decimals=0)
    df_predicted[:, 3] = np.round(df_predicted[:, 3], decimals=0)
    df_predicted = pd.DataFrame(df_predicted)
    df_predicted.columns = df_main_main_full.columns
    
    ape_i = []
    for i in I[::steps]: 
        df_predicted_i_filtered = df_predicted[df_predicted['i'] == i]
        meow = df_main_main_full[df_main_main_full['radius'] == R_validation]
        df_actual_i_filtered = meow[meow['i'] == i]
            
        df_predicted_contour = pd.DataFrame()
        df_actual_contour = pd.DataFrame()
        
        ape_n = []
        
        for n in N[::steps]:
            df_predicted_i_n_filtered = df_predicted_i_filtered[df_predicted_i_filtered['n'] == n]
            df_actual_i_n_filtered = df_actual_i_filtered[df_actual_i_filtered['n'] == n]
            
            e_actual = pd.Series(df_actual_i_n_filtered['e'])
            e_predicted = pd.Series(df_predicted_i_n_filtered['e'])           
            e_actual = e_actual.reset_index(drop=True)
            e_predicted = e_predicted.reset_index(drop=True)
            
            error = e_actual - e_predicted
            abs_error = error.abs()*1000
            percentage_error = (error.abs() / e_actual) * 100
            ape = percentage_error.mean()
            ape_n.append(ape)
            
            fig = plt.figure(figsize=(7, 4))
            ax = plt.axes()
            ax.plot(df_actual_i_n_filtered['y'], e_actual, color = "blue")
            scatter = ax.scatter(df_predicted_i_n_filtered['y'], e_predicted, s = 10, c=abs_error, cmap = "twilight_shifted", alpha = 1)
            clb=fig.colorbar(scatter, ticks=(np.around(np.linspace(min(abs_error), max(abs_error), num=6), decimals=5)).tolist())
            clb.ax.set_title('Absolute Error (10\u207B\u00B3)\n', fontweight="bold", fontsize = 15)
            for l in clb.ax.yaxis.get_ticklabels():
                l.set_weight("bold")
                l.set_fontsize(15)
            #graph formatting     
            ax.tick_params(which='major', width=2.00)
            ax.tick_params(which='minor', width=2.00)
            ax.xaxis.label.set_fontsize(15)
            ax.xaxis.label.set_weight("bold")
            ax.yaxis.label.set_fontsize(15)
            ax.yaxis.label.set_weight("bold")
            ax.tick_params(axis='both', which='major', labelsize=15)
            ax.set_yticklabels(ax.get_yticks(), weight='bold')
            ax.set_xticklabels(ax.get_xticks(), weight='bold')
            ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.3f}'))
            ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.spines['bottom'].set_linewidth(2)
            ax.spines['left'].set_linewidth(2)
            
            
            plt.xlabel("x-position (µm)")
            plt.ylabel("E-Field (eV)")
            plt.legend(["Actual", "Prediction"], prop={'weight': 'bold','size': 10}, loc = "upper left")
            plt.title("Actual/Prediction\n"+"n="+str(n)+"\n"+"i="+str(i)+"\n\n", fontweight = 'bold')
            plt.show()
            plt.close()
            
            df_predicted_contour[n] = e_predicted
            df_actual_contour[n] = e_actual
        x = np.linspace(0, 50, num=df_actual_contour.shape[1])
        y = np.linspace(-25, 25, num=df_actual_contour.shape[0])
        colorbarmax = df_actual_contour.max().max()
            
        X,Y = np.meshgrid(x,y)
        fig = plt.figure(figsize=(5, 4))
        ax = plt.axes()
        cp=ax.contourf(X,Y,df_actual_contour, 200, zdir='z', offset=-100, cmap='twilight_shifted')
        clb=fig.colorbar(cp, ticks=(np.around(np.linspace(0.0, colorbarmax, num=6), decimals=3)).tolist())
        clb.ax.set_title('Electric Field (eV)', fontweight="bold")
        for l in clb.ax.yaxis.get_ticklabels():
            l.set_weight("bold")
            l.set_fontsize(15)
        ax.set_xlabel('x-position (µm)', fontsize=20, fontweight="bold", labelpad=1)
        ax.set_ylabel('y-position (µm)', fontsize=20, fontweight="bold", labelpad=1)
        ax.xaxis.label.set_fontsize(20)
        ax.xaxis.label.set_weight("bold")
        ax.yaxis.label.set_fontsize(20)
        ax.yaxis.label.set_weight("bold")
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.set_yticklabels(ax.get_yticks(), weight='bold')
        ax.set_xticklabels(ax.get_xticks(), weight='bold')
        ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
        ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
        plt.title("Actual Beam (i = "+str(i)+")\n", fontweight = 'bold', fontsize=15)
        plt.show()
        plt.close()
        
        time.sleep(2)
        
        colorbarmax = df_predicted_contour.max().max()
            
        X,Y = np.meshgrid(x,y)
        fig = plt.figure(figsize=(5, 4))
        ax = plt.axes()
        cp=ax.contourf(X,Y,df_predicted_contour, 200, zdir='z', offset=-100, cmap='twilight_shifted')
        clb=fig.colorbar(cp, ticks=(np.around(np.linspace(0.0, colorbarmax, num=6), decimals=3)).tolist())
        clb.ax.set_title('Electric Field (eV)', fontweight="bold")
        for l in clb.ax.yaxis.get_ticklabels():
            l.set_weight("bold")
            l.set_fontsize(15)
        ax.set_xlabel('x-position (µm)', fontsize=20, fontweight="bold", labelpad=1)
        ax.set_ylabel('y-position (µm)', fontsize=20, fontweight="bold", labelpad=1)
        ax.xaxis.label.set_fontsize(20)
        ax.xaxis.label.set_weight("bold")
        ax.yaxis.label.set_fontsize(20)
        ax.yaxis.label.set_weight("bold")
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.set_yticklabels(ax.get_yticks(), weight='bold')
        ax.set_xticklabels(ax.get_xticks(), weight='bold')
        ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
        ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
        plt.title("Actual Beam (i = "+str(i)+")\n", fontweight = 'bold', fontsize=15)
        plt.show()
        plt.close()
        
        time.sleep(2)
    
    
    
        ape_n_mean = sum(ape_n)/len(ape_n)
        ape_i.append(ape_n_mean)
        
if Plot_selected_i == True:
    steps = 2
    import tensorflow as tf
    
    min_index = df_optimization['ape_list'].idxmin()
    layer_size_op = df_optimization.iloc[min_index, :]
    layer_size_op = int(layer_size_op['nodes']) 
    dense_layer_op = df_optimization.iloc[min_index, :]
    dense_layer_op = int(dense_layer_op['layers']) 
    
    I = np.arange(100, 400, 100)
    
    # Load the saved model
    model = tf.keras.models.load_model(file_path+"3dmultisinglebeam_model_nodes"+str(layer_size_op)+"_layers"+str(dense_layer_op))
    
    df_main_main_full = pd.DataFrame()
    
    for r in R:
        file = "grating012umpitch05dutycycle"+str(r)+"um"
        hdf5_file = file_path2+file+".h5"
        # Load the h5 file
        with h5py.File(hdf5_file, 'r') as f:
            # Get the dataset
            dset = f[file]
            # Load the dataset into a numpy array
            arr_3d_loaded = dset[()]
        df_main_main = pd.DataFrame()
        for i in I:  
            N = np.arange(0,df_df.shape[1],1)
            
            df_df = arr_3d_loaded[:,:,i]
            df_df = pd.DataFrame(df_df)
        
            df_main = pd.DataFrame() 
            for n in N[::steps]:
                x1 = np.linspace(0, 50, num=df_df.shape[1])[::steps]
                y = np.linspace(-25, 25, num =df_df.shape[0])[::steps]
                df = pd.DataFrame()
                df['y'] = y
                df['radius'] = r
                df['n'] = n
                df['i'] = i
                e = df_df[n]
                e = e[::steps]
                e.reset_index(drop=True, inplace=True)
                df['e'] = e
                df_main = pd.concat([df_main, df], axis= 0)
                print("{}-radius-{}-n-{}-i".format(r, n, i))
            df_main_main = pd.concat([df_main_main, df_main], axis= 0)
        df_main_main_full = pd.concat([df_main_main_full, df_main_main], axis= 0)
    
            
    scaler = MinMaxScaler()
    df_main_main_full_norm = scaler.fit_transform(df_main_main_full)
    df_main_main_full_norm = pd.DataFrame(df_main_main_full_norm)
    df_main_main_full_norm = df_main_main_full_norm.round(5)
    
    df_training_norm = pd.DataFrame()
    for r in R_training_norm:
        df_intermediate = df_main_main_full_norm[df_main_main_full_norm[1] == r]
        df_training_norm = pd.concat([df_training_norm, df_intermediate], axis= 0)
    
    df_validation_norm = df_main_main_full_norm[df_main_main_full_norm[1] == R_validation_norm]
    
    
    X_train = df_training_norm.iloc[:, 0:-1]
    X_train.reset_index(drop=True, inplace=True)
    #X_train = X_train.values.astype(float)
    
    y_train = df_training_norm.iloc[:, -1]
    y_train = pd.DataFrame(y_train)
    y_train.reset_index(drop=True, inplace=True)
    
    X_test = df_validation_norm.iloc[:, 0:-1]
    X_test.reset_index(drop=True, inplace=True)
    #X_test = X_test.values.astype(float)
    
    y_test = df_validation_norm.iloc[:, -1]
    y_test = pd.DataFrame(y_test)
    y_test.reset_index(drop=True, inplace=True)
    
    predictions = model.predict(X_test)
    
    X_test.reset_index(drop=True, inplace=True)
    predictions = pd.DataFrame(predictions)
    predictions.reset_index(drop=True, inplace=True)
    df_predicted_norm = pd.concat([X_test, predictions], axis= 1)
    
    df_predicted = scaler.inverse_transform(df_predicted_norm)
    df_predicted[:, 2] = np.round(df_predicted[:, 2], decimals=0)
    df_predicted[:, 3] = np.round(df_predicted[:, 3], decimals=0)
    df_predicted = pd.DataFrame(df_predicted)
    df_predicted.columns = df_main_main_full.columns
    
    ape_i = []
    for i in I: 
        df_predicted_i_filtered = df_predicted[df_predicted['i'] == i]
        meow = df_main_main_full[df_main_main_full['radius'] == R_validation]
        df_actual_i_filtered = meow[meow['i'] == i]
            
        df_predicted_contour = pd.DataFrame()
        df_actual_contour = pd.DataFrame()
        
        ape_n = []
        
        for n in N[::steps]:
            df_predicted_i_n_filtered = df_predicted_i_filtered[df_predicted_i_filtered['n'] == n]
            df_actual_i_n_filtered = df_actual_i_filtered[df_actual_i_filtered['n'] == n]
            
            e_actual = pd.Series(df_actual_i_n_filtered['e'])
            e_predicted = pd.Series(df_predicted_i_n_filtered['e'])           
            e_actual = e_actual.reset_index(drop=True)
            e_predicted = e_predicted.reset_index(drop=True)
            
            error = e_actual - e_predicted
            abs_error = error.abs()*1000
            percentage_error = (error.abs() / e_actual) * 100
            ape = percentage_error.mean()
            ape_n.append(ape)
            
            fig = plt.figure(figsize=(7, 4))
            ax = plt.axes()
            ax.plot(df_actual_i_n_filtered['y'], e_actual, color = "blue")
            scatter = ax.scatter(df_predicted_i_n_filtered['y'], e_predicted, s = 10, c=abs_error, cmap = "twilight_shifted", alpha = 1)
            clb=fig.colorbar(scatter, ticks=(np.around(np.linspace(min(abs_error), max(abs_error), num=6), decimals=5)).tolist())
            clb.ax.set_title('Absolute Error (10\u207B\u00B3)\n', fontweight="bold", fontsize = 15)
            for l in clb.ax.yaxis.get_ticklabels():
                l.set_weight("bold")
                l.set_fontsize(15)
            #graph formatting     
            ax.tick_params(which='major', width=2.00)
            ax.tick_params(which='minor', width=2.00)
            ax.xaxis.label.set_fontsize(15)
            ax.xaxis.label.set_weight("bold")
            ax.yaxis.label.set_fontsize(15)
            ax.yaxis.label.set_weight("bold")
            ax.tick_params(axis='both', which='major', labelsize=15)
            ax.set_yticklabels(ax.get_yticks(), weight='bold')
            ax.set_xticklabels(ax.get_xticks(), weight='bold')
            ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.3f}'))
            ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.spines['bottom'].set_linewidth(2)
            ax.spines['left'].set_linewidth(2)
            
            
            plt.xlabel("x-position (µm)")
            plt.ylabel("E-Field (eV)")
            plt.legend(["Actual", "Prediction"], prop={'weight': 'bold','size': 10}, loc = "upper left")
            plt.title("Actual/Prediction\n"+"n="+str(n)+"\n"+"i="+str(i)+"\n\n", fontweight = 'bold')
            plt.show()
            plt.close()
            
            df_predicted_contour[n] = e_predicted
            df_actual_contour[n] = e_actual
        x = np.linspace(0, 50, num=df_actual_contour.shape[1])
        y = np.linspace(-25, 25, num=df_actual_contour.shape[0])
        colorbarmax = df_actual_contour.max().max()
            
        X,Y = np.meshgrid(x,y)
        fig = plt.figure(figsize=(5, 4))
        ax = plt.axes()
        cp=ax.contourf(X,Y,df_actual_contour, 200, zdir='z', offset=-100, cmap='twilight_shifted')
        clb=fig.colorbar(cp, ticks=(np.around(np.linspace(0.0, colorbarmax, num=6), decimals=3)).tolist())
        clb.ax.set_title('Electric Field (eV)', fontweight="bold")
        for l in clb.ax.yaxis.get_ticklabels():
            l.set_weight("bold")
            l.set_fontsize(15)
        ax.set_xlabel('x-position (µm)', fontsize=20, fontweight="bold", labelpad=1)
        ax.set_ylabel('y-position (µm)', fontsize=20, fontweight="bold", labelpad=1)
        ax.xaxis.label.set_fontsize(20)
        ax.xaxis.label.set_weight("bold")
        ax.yaxis.label.set_fontsize(20)
        ax.yaxis.label.set_weight("bold")
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.set_yticklabels(ax.get_yticks(), weight='bold')
        ax.set_xticklabels(ax.get_xticks(), weight='bold')
        ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
        ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
        plt.title("Actual Beam (i = "+str(i)+")\n", fontweight = 'bold', fontsize=15)
        plt.show()
        plt.close()
        
        time.sleep(2)
        
        colorbarmax = df_predicted_contour.max().max()
            
        X,Y = np.meshgrid(x,y)
        fig = plt.figure(figsize=(5, 4))
        ax = plt.axes()
        cp=ax.contourf(X,Y,df_predicted_contour, 200, zdir='z', offset=-100, cmap='twilight_shifted')
        clb=fig.colorbar(cp, ticks=(np.around(np.linspace(0.0, colorbarmax, num=6), decimals=3)).tolist())
        clb.ax.set_title('Electric Field (eV)', fontweight="bold")
        for l in clb.ax.yaxis.get_ticklabels():
            l.set_weight("bold")
            l.set_fontsize(15)
        ax.set_xlabel('x-position (µm)', fontsize=20, fontweight="bold", labelpad=1)
        ax.set_ylabel('y-position (µm)', fontsize=20, fontweight="bold", labelpad=1)
        ax.xaxis.label.set_fontsize(20)
        ax.xaxis.label.set_weight("bold")
        ax.yaxis.label.set_fontsize(20)
        ax.yaxis.label.set_weight("bold")
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.set_yticklabels(ax.get_yticks(), weight='bold')
        ax.set_xticklabels(ax.get_xticks(), weight='bold')
        ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
        ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
        plt.title("Actual Beam (i = "+str(i)+")\n", fontweight = 'bold', fontsize=15)
        plt.show()
        plt.close()
        
        time.sleep(2)
    
    
    
        ape_n_mean = sum(ape_n)/len(ape_n)
        ape_i.append(ape_n_mean)
if Plot_ape == True:
    df_ape = pd.read_csv("C:\\Users\\limyu\\Google Drive\\3D Data ML Beam\\ape.csv")
    ape_plot = pd.Series(df_ape['ape'])
    z = np.linspace(-5, 45, num =317)
    begin_z = z[100]
    end_z = z[300]
    z_plot = np.linspace(begin_z, end_z, num=len(ape_plot))
    
    color = []
    for _ in range(5):
        color.append('red')
        for _ in range(19):
            color.append('black')
    color.append('red')
    
    S = []
    for _ in range(5):
        S.append(50)
        for _ in range(19):
            S.append(10)
    S.append(50)
    
    
    
    fig = plt.figure(figsize=(7, 4))
    ax = plt.axes()
    ax.scatter(z_plot, ape_plot, s = S, color = color, alpha = 1)
    
    #graph formatting     
    ax.tick_params(which='major', width=2.00)
    ax.tick_params(which='minor', width=2.00)
    ax.xaxis.label.set_fontsize(15)
    ax.xaxis.label.set_weight("bold")
    ax.yaxis.label.set_fontsize(15)
    ax.yaxis.label.set_weight("bold")
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.set_yticklabels(ax.get_yticks(), weight='bold')
    ax.set_xticklabels(ax.get_xticks(), weight='bold')
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    
    
    plt.xlabel("z-position (µm)")
    plt.ylabel("APE (%)")
    #plt.legend(["Actual", "Prediction"], prop={'weight': 'bold','size': 10}, loc = "best")
    plt.title("Actual/Prediction\n"+"n="+str(n)+"\n"+"i="+str(i)+"\n\n", fontweight = 'bold')
    plt.show()
    plt.close()
    
