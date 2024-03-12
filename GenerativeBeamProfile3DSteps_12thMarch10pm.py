# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 09:59:37 2024

@author: ADMIN
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 23:23:12 2024

@author: ADMIN
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

file_list = "grating12pitch100", "grating12_11pitch6_4", "grating012umpitch05dutycycle60um"
file_path = "C:\\Users\\ADMIN\\Downloads\\3dpredictiondata\\"
file_path2 = "C:\\Users\\ADMIN\\Downloads\\"
steps = 10
df_main = pd.DataFrame()

R = 20, 30, 40, 50, 60
R = pd.Series(R)
R_norm = (R - min(R))/(max(R) - min(R))
R_norm = round(R_norm, 5)
R_validation = 50
R_prediction = 30
R_validation_norm = round((R_validation - min(R))/(max(R) - min(R)), 5)
R_prediction_norm = round((R_prediction - min(R))/(max(R) - min(R)), 5)
R_training = R[R != R_validation]
R_training = R_training[R_training != R_prediction]
R_training_norm = R_norm[R_norm != R_validation_norm]
R_training_norm = R_training_norm[R_training_norm != R_prediction_norm]
R_training_norm = round(R_training_norm, 5)

i = 316
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
    cp=ax.contourf(X,Y,df_df, 200, zdir='z', offset=-100, cmap='viridis')
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


dense_layers = [3,4,5,6,8]
layer_sizes = [6,8,10,15,20]

name = []
mae = []
nodes = []
layers1=[]
ape_list = []
train_test_ape = []
RMS = []
mae_list = []
train_test_mae = []
time_list = []

df_training_validation = pd.DataFrame()

for dense_layer in dense_layers:
        for layer_size in layer_sizes:
            start_time = time.time()

            NAME = "{}-nodes-{}-dense-{}".format(layer_size, dense_layer, int(time.time()))
            print(NAME)
            name.append(NAME)
            nodes.append(layer_size)
            layers1.append(dense_layer)
            
            N = np.arange(0,df_df.shape[1],1)
            df_main_main = pd.DataFrame()
            
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
                df_df = pd.DataFrame(df_df)
                
                df_main = pd.DataFrame() 
                for n in N[::steps]:
                    x1 = np.linspace(0, 50, num=df_df.shape[1])[::steps]
                    y = np.linspace(-25, 25, num =df_df.shape[0])[::steps]
                    df = pd.DataFrame()
                    df['y'] = y
                    df['radius'] = r
                    df['n'] = n
                    e = df_df[n]
                    e = e[::steps]
                    e.reset_index(drop=True, inplace=True)
                    df['e'] = e
                    df_main = pd.concat([df_main, df], axis= 0)
                    print("{}-radius-{}-n".format(r, n))
                df_main_main = pd.concat([df_main_main, df_main], axis= 0)
            
            



                
            scaler = MinMaxScaler()
            df_main_main_norm = scaler.fit_transform(df_main_main)
            df_main_main_norm = pd.DataFrame(df_main_main_norm)
            df_main_main_norm = df_main_main_norm.round(5)
            
            df_training_norm = pd.DataFrame()
            for r in R_training_norm:
                df_intermediate = df_main_main_norm[df_main_main_norm[1] == r]
                df_training_norm = pd.concat([df_training_norm, df_intermediate], axis= 0)
            
            df_validation_norm = df_main_main_norm[df_main_main_norm[1] == R_validation_norm]

            X_train = df_training_norm.iloc[:, 0:-1]
            #X_train = X_train.values.astype(float)

            y_train = df_training_norm.iloc[:, -1]
            y_train = pd.DataFrame(y_train)

            X_test = df_validation_norm.iloc[:, 0:-1]
            #X_test = X_test.values.astype(float)

            y_test = df_validation_norm.iloc[:, -1]
            y_test = pd.DataFrame(y_test)
            
            model = Sequential()
            model.add(Dense(len(X_train.keys()),  input_shape=[len(X_train.keys())]))
            for _ in range(dense_layer):
                    model.add(Dense(layer_size))
                    model.add(Activation('elu'))
                    #layer_size = int(round(layer_size*0.9, 0))

            model.add(Dense(y_train.shape[1]))


            # Compile the model
            model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
            
            history = model.fit(X_train, y_train, epochs=100,validation_data=(X_test, y_test), batch_size = 10)
            
            
            # Evaluate the model
            loss1, mae1 = model.evaluate(X_test, y_test)
            loss2, mae2 = model.evaluate(X_train, y_train)
            
            # Print the results
            print('Mean Absolute Error:', mae1)
            print('Training Loss', loss2)
            print('Validation loss', loss1)
            print(history.history['loss'])


            training_loss = pd.Series(history.history['loss'])
            validation_loss = pd.Series(history.history['val_loss'])
            
            df_training_validation['T'+'_N'+str(layer_size)+'_L'+str(dense_layer)] = training_loss
            df_training_validation['V'+'_N'+str(layer_size)+'_L'+str(dense_layer)] = validation_loss

            rms = calculate_rms_error(validation_loss[50:])

            
            diff = (validation_loss[50:] - training_loss[50:]).abs()
            rel_error = diff / training_loss
            pct_error = rel_error * 100
            ape = pct_error.mean()
            train_test_ape.append(ape)
            
            mae = mean_absolute_error(validation_loss[50:], training_loss[50:])
            train_test_mae.append(mae)

            
            epochs = range(1, 100 + 1)
            
            fig = plt.figure(figsize=(7, 4))
            ax = plt.axes()
            ax.plot(epochs, training_loss, color = "blue")
            ax.plot(epochs, validation_loss, color = "red")
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
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.legend(["Training_loss", "Validation Loss"], prop={'weight': 'bold','size': 10}, loc = "upper left")
            plt.title("Training/Validation Loss\n"+NAME, fontweight = 'bold')
            plt.show()
            plt.close()
            
            predictions = model.predict(X_test)
            
            X_test = pd.DataFrame(X_test)
            X_test.reset_index(drop=True, inplace=True)
            predictions = pd.DataFrame(predictions)
            predictions.reset_index(drop=True, inplace=True)
            
            
            df_predicted_norm = pd.concat([X_test, predictions], axis= 1)
            
            df_predicted = scaler.inverse_transform(df_predicted_norm)
            df_predicted[:, 2] = np.round(df_predicted[:, 2], decimals=0)
            
            df_predicted_contour = pd.DataFrame()
            
            for n in N[::steps]:
                df_predicted_n_filtered = df_predicted[df_predicted[:, 2] == n]
                df_predicted_n_filtered = pd.DataFrame(df_predicted_n_filtered )
                df_predicted_n_filtered.columns = [df_main_main.columns]
                
                meow = df_main_main[df_main_main['radius'] == R_validation]
                df_actual_n_filtered = meow[meow['n'] == n]

            
          
                fig = plt.figure(figsize=(7, 4))
                ax = plt.axes()
                ax.plot(df_actual_n_filtered['y'], df_actual_n_filtered['e'], color = "blue")
                ax.plot(df_predicted_n_filtered['y'], df_predicted_n_filtered['e'], color = "red")
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
                plt.title("Actual/Prediction\n"+NAME, fontweight = 'bold')
                plt.show()
                plt.close()
                
                df_predicted_contour[n] = df_predicted_n_filtered['e']
            
            df_predicted_contour.to_csv(file_path+"singlebeam_model_nodes"+str(layer_size)+"_layers"+str(dense_layer)+".csv")
            model.save(file_path+"singlebeam_model_nodes"+str(layer_size)+"_layers"+str(dense_layer))
            
            df_actual_benchmark = df_main_main[df_main_main['radius'] == R_validation]
            
            ape_n = []
            mae_n = []
            
            for n in N[::steps]:
                df_actual_benchmark_n_filtered = df_actual_benchmark[df_actual_benchmark['n'] ==n] 
                e_actual = pd.Series(df_actual_benchmark_n_filtered['e'])
                e_predicted = pd.Series(df_predicted_contour[n])
                
                error = e_actual - e_predicted
                percentage_error = (error.abs() / e_actual) * 100
                ape = percentage_error.mean()
                ape_n.append(ape)
                
                mae = mean_absolute_error(e_actual, e_predicted)
                mae_n.append(mae)
            
            ape_total = sum(ape_n)/len(ape_n)
            ape_list.append(ape_total)
            mae_total = sum(mae_n)/len(mae_n)
            mae_list.append(mae_total)
            
            end_time = time.time()
            time_consumned = end_time - start_time
            time_list.append(time_consumned)
            
            
df_result = pd.DataFrame()
df_result['nodes'] = nodes
df_result['layers'] = layers1
df_result['ape'] = ape_list
df_result['maee'] = mae_list
df_result['train_test_ape'] = train_test_ape
df_result['train_test_mae'] = train_test_mae
df_result['time_list'] = time_list

min_index = df_result['ape'].idxmin()
dense_layer_op = df_result.iloc[min_index, 1]
layer_size_op = df_result.iloc[min_index, 0]

df_result.to_csv(file_path+"singlebeam_model_optimization_result.csv")
df_training_validation.to_csv(file_path+"singlebeam_df_training_validation.csv")

z_cutoff = 100
I = np.arange(z_cutoff,arr_3d_loaded.shape[2],1)

df_training_validation_full = pd.DataFrame()
train_test_ape = []
train_test_mae = []  
ape_list = []
mae_list = []
time_list = []
i_list = []


for i in I[::steps]:  
    i_list.append(i)
    N = np.arange(0,df_df.shape[1],1)
    df_main_main = pd.DataFrame()
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
        df_df = pd.DataFrame(df_df)
        
        df_main = pd.DataFrame() 
        for n in N[::steps]:
            x1 = np.linspace(0, 50, num=df_df.shape[1])[::steps]
            y = np.linspace(-25, 25, num =df_df.shape[0])[::steps]
            df = pd.DataFrame()
            df['y'] = y
            df['radius'] = r
            df['n'] = n
            e = df_df[n]
            e = e[::steps]
            e.reset_index(drop=True, inplace=True)
            df['e'] = e
            df_main = pd.concat([df_main, df], axis= 0)
            print("{}-radius-{}-n".format(r, n))
        df_main_main = pd.concat([df_main_main, df_main], axis= 0)
        
    scaler = MinMaxScaler()
    df_main_main_norm = scaler.fit_transform(df_main_main)
    df_main_main_norm = pd.DataFrame(df_main_main_norm)
    df_main_main_norm = df_main_main_norm.round(5)
    
    df_training_norm = pd.DataFrame()
    for r in R_training_norm:
        df_intermediate = df_main_main_norm[df_main_main_norm[1] == r]
        df_training_norm = pd.concat([df_training_norm, df_intermediate], axis= 0)
    
    df_validation_norm = df_main_main_norm[df_main_main_norm[1] == R_validation_norm]

    X_train = df_training_norm.iloc[:, 0:-1]
    #X_train = X_train.values.astype(float)

    y_train = df_training_norm.iloc[:, -1]
    y_train = pd.DataFrame(y_train)

    X_test = df_validation_norm.iloc[:, 0:-1]
    #X_test = X_test.values.astype(float)

    y_test = df_validation_norm.iloc[:, -1]
    y_test = pd.DataFrame(y_test)
    
    model = Sequential()
    model.add(Dense(len(X_train.keys()),  input_shape=[len(X_train.keys())]))
    for _ in range(dense_layer_op):
            model.add(Dense(layer_size_op))
            model.add(Activation('elu'))
            #layer_size = int(round(layer_size*0.9, 0))

    model.add(Dense(y_train.shape[1]))


    # Compile the model
    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
    
    
    start_time = time.time()
    history = model.fit(X_train, y_train, epochs=100,validation_data=(X_test, y_test), batch_size = 10)
    end_time = time.time()
    
    time_consumed = end_time - start_time
    time_list.append(time_consumed)
    
    # Evaluate the model
    loss1, mae1 = model.evaluate(X_test, y_test)
    loss2, mae2 = model.evaluate(X_train, y_train)
    
    # Print the results
    print('Mean Absolute Error:', mae1)
    print('Training Loss', loss2)
    print('Validation loss', loss1)
    print(history.history['loss'])


    training_loss = pd.Series(history.history['loss'])
    validation_loss = pd.Series(history.history['val_loss'])
    
    df_training_validation_full['T'+'_N'+str(layer_size_op)+'_L'+str(dense_layer_op)+'_i'+str(i)] = training_loss
    df_training_validation_full['V'+'_N'+str(layer_size_op)+'_L'+str(dense_layer_op)+'_i'+str(i)] = validation_loss

    rms = calculate_rms_error(validation_loss[50:])

    
    diff = (validation_loss[50:] - training_loss[50:]).abs()
    rel_error = diff / training_loss
    pct_error = rel_error * 100
    ape = pct_error.mean()
    train_test_ape.append(ape)
    
    mae = mean_absolute_error(validation_loss[50:], training_loss[50:])
    train_test_mae.append(mae)

    
    epochs = range(1, 100 + 1)
    
    fig = plt.figure(figsize=(7, 4))
    ax = plt.axes()
    ax.plot(epochs, training_loss, color = "blue")
    ax.plot(epochs, validation_loss, color = "red")
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
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(["Training_loss", "Validation Loss"], prop={'weight': 'bold','size': 10}, loc = "upper left")
    plt.title("Training/Validation Loss\n"+NAME, fontweight = 'bold')
    plt.show()
    plt.close()
    
    predictions = model.predict(X_test)
    
    X_test = pd.DataFrame(X_test)
    X_test.reset_index(drop=True, inplace=True)
    predictions = pd.DataFrame(predictions)
    predictions.reset_index(drop=True, inplace=True)
    
    
    df_predicted_norm = pd.concat([X_test, predictions], axis= 1)
    
    df_predicted = scaler.inverse_transform(df_predicted_norm)
    df_predicted[:, 2] = np.round(df_predicted[:, 2], decimals=0)
    
    df_predicted_contour = pd.DataFrame()
    
    for n in N[::steps]:
        df_predicted_n_filtered = df_predicted[df_predicted[:, 2] == n]
        df_predicted_n_filtered = pd.DataFrame(df_predicted_n_filtered )
        df_predicted_n_filtered.columns = [df_main_main.columns]
        
        meow = df_main_main[df_main_main['radius'] == R_validation]
        df_actual_n_filtered = meow[meow['n'] == n]

    
  
        fig = plt.figure(figsize=(7, 4))
        ax = plt.axes()
        ax.plot(df_actual_n_filtered['y'], df_actual_n_filtered['e'], color = "blue")
        ax.plot(df_predicted_n_filtered['y'], df_predicted_n_filtered['e'], color = "red")
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
        
        df_predicted_contour[n] = df_predicted_n_filtered['e']
    
    df_predicted_contour.to_csv(file_path+"2dmodel_nodes"+str(layer_size_op)+"_layers"+str(dense_layer_op)+"_i"+str(i)+".csv")
    model.save(file_path+"2dmodel_nodes"+str(layer_size_op)+"_layers"+str(dense_layer_op)+"_i"+str(i))
    
    df_actual_benchmark = df_main_main[df_main_main['radius'] == R_validation]
    
    ape_n = []
    mae_n = []
    
    for n in N[::steps]:
        df_actual_benchmark_n_filtered = df_actual_benchmark[df_actual_benchmark['n'] ==n] 
        e_actual = pd.Series(df_actual_benchmark_n_filtered['e'])
        e_predicted = pd.Series(df_predicted_contour[n])
        
        error = e_actual - e_predicted
        percentage_error = (error.abs() / e_actual) * 100
        ape = percentage_error.mean()
        ape_n.append(ape)
        
        mae = mean_absolute_error(e_actual, e_predicted)
        mae_n.append(mae)
    
    ape_total = sum(ape_n)/len(ape_n)
    ape_list.append(ape_total)
    mae_total = sum(mae_n)/len(mae_n)
    mae_list.append(mae_total)
    

df_result_full = pd.DataFrame()
df_result_full['i_list'] = i_list
df_result_full['train_test_ape'] = train_test_ape
df_result_full['train_test_mae'] = train_test_mae
df_result_full['ape_list'] = ape_list
df_result_full['mae_list'] = mae_list
df_result_full['time_list'] = time_list


df_result_full.to_csv(file_path+"2dmodel_optimization_result.csv")
df_training_validation_full.to_csv(file_path+"2ddf_training_validation_full.csv")
    
    
#========================================================================================


z_cutoff = 100
I = np.arange(z_cutoff,arr_3d_loaded.shape[2],1)

df_training_validation_full = pd.DataFrame()
train_test_ape = []
train_test_mae = []  
ape_list = []
mae_list = []
time_list = []
i_list = []

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
        i_list.append(i)
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

model = Sequential()
model.add(Dense(len(X_train.keys()),  input_shape=[len(X_train.keys())]))
for _ in range(dense_layer_op):
        model.add(Dense(layer_size_op))
        model.add(Activation('elu'))
        #layer_size = int(round(layer_size*0.9, 0))

model.add(Dense(y_train.shape[1]))


# Compile the model
model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])


start_time = time.time()
history = model.fit(X_train, y_train, epochs=100,validation_data=(X_test, y_test), batch_size = 10)
end_time = time.time()

time_consumed = end_time - start_time
#time_list.append(time_consumed)

# Evaluate the model
loss1, mae1 = model.evaluate(X_test, y_test)
loss2, mae2 = model.evaluate(X_train, y_train)

# Print the results
print('Mean Absolute Error:', mae1)
print('Training Loss', loss2)
print('Validation loss', loss1)
print(history.history['loss'])


training_loss = pd.Series(history.history['loss'])
validation_loss = pd.Series(history.history['val_loss'])

df_training_validation_full['T'+'_N'+str(layer_size_op)+'_L'+str(dense_layer_op)+'_i'+str(i)] = training_loss
df_training_validation_full['V'+'_N'+str(layer_size_op)+'_L'+str(dense_layer_op)+'_i'+str(i)] = validation_loss

rms = calculate_rms_error(validation_loss[50:])


diff = (validation_loss[50:] - training_loss[50:]).abs()
rel_error = diff / training_loss
pct_error = rel_error * 100
ape = pct_error.mean()
#train_test_ape.append(ape)

mae = mean_absolute_error(validation_loss[50:], training_loss[50:])
#train_test_mae.append(mae)


epochs = range(1, 100 + 1)

fig = plt.figure(figsize=(7, 4))
ax = plt.axes()
ax.plot(epochs, training_loss, color = "blue")
ax.plot(epochs, validation_loss, color = "red")
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
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(["Training_loss", "Validation Loss"], prop={'weight': 'bold','size': 10}, loc = "upper left")
plt.title("Training/Validation Loss\n"+NAME, fontweight = 'bold')
plt.show()
plt.close()

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

for i in I[::steps]: 
    df_predicted_i_filtered = df_predicted[df_predicted['i'] == i]
    meow = df_main_main_full[df_main_main_full['radius'] == R_validation]
    df_actual_i_filtered = meow[meow['i'] == i]
        
    df_predicted_contour = pd.DataFrame()
    
    for n in N[::steps]:
        df_predicted_i_n_filtered = df_predicted_i_filtered[df_predicted_i_filtered['n'] == n]
        df_actual_i_n_filtered = df_actual_i_filtered[df_actual_i_filtered['n'] == n]
    
        fig = plt.figure(figsize=(7, 4))
        ax = plt.axes()
        ax.plot(df_actual_i_n_filtered['y'], df_actual_i_n_filtered['e'], color = "blue")
        ax.plot(df_predicted_i_n_filtered['y'], df_predicted_i_n_filtered['e'], color = "red")
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
        
        df_predicted_contour[n] = df_predicted_n_filtered['e']
    fig = plt.figure(figsize=(7, 4))
    ax = plt.axes()
    plt.show()
    plt.close()
    
    df_predicted_contour.to_csv(file_path+"3dmodel_nodes"+str(layer_size_op)+"_layers"+str(dense_layer_op)+"_i"+str(i)+"_radius"+str(R_validation)+".csv")
model.save(file_path+"3dmodel_nodes"+str(layer_size_op)+"_layers"+str(dense_layer_op)+"_i"+str(i))
