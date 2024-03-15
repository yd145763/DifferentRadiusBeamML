# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 00:09:16 2024

@author: ADMIN
"""

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
file_path = "/home/grouptan/Documents/yudian/DNNPredictBeamData/"
file_path2 = "/home/grouptan/Documents/yudian/DNNPredictBeamSource/"
steps = 20
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
Data_Source_Verification = False

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
        cp=ax.contourf(X,Y,df_main_contour, 200, zdir='z', offset=-100, cmap='viridis')
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


dense_layers = [6,8,10,12,15]
layer_sizes = [10,15,20,25,30]

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

training_steps = 20

df_training_validation = pd.DataFrame()
df_results = pd.DataFrame()

for dense_layer in dense_layers:
        for layer_size in layer_sizes:
            start_time = time.time()

            NAME = "{}-nodes-{}-dense-{}".format(layer_size, dense_layer, int(time.time()))
            print(NAME)
            name.append(NAME)
            nodes.append(layer_size)
            layers1.append(dense_layer)
            
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
                
                z_cutoff = 100
                I = np.arange(z_cutoff,arr_3d_loaded.shape[2],1)
                
                for i in I[::training_steps]:  
                    N = np.arange(0,df_df.shape[1],1)
                    
                    df_df = arr_3d_loaded[:,:,i]
                    df_df = pd.DataFrame(df_df)
                
                    df_main = pd.DataFrame() 
                    for n in N[::training_steps]:
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
                df_intermediate = df_main_main_full_norm [df_main_main_full_norm[1] == r]
                df_training_norm = pd.concat([df_training_norm, df_intermediate], axis= 0)
            
            df_validation_norm = df_main_main_full_norm[df_main_main_full_norm[1] == R_validation_norm]

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
            
            start_time = time.time()
            
            history = model.fit(X_train, y_train, epochs=100,validation_data=(X_test, y_test), batch_size = 10)
            
            end_time = time.time()
            training_time = end_time - start_time
            
            time_list.append(training_time)
            
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
            df_predicted[:, 3] = np.round(df_predicted[:, 3], decimals=0)
            df_predicted = pd.DataFrame(df_predicted)
            df_predicted.columns = df_main_main_full.columns
            
            ape_n_mean_i_mean = []
            mae_n_mean_i_mean = []
            for i in I[::training_steps]: 
                df_predicted_i_filtered = df_predicted[df_predicted['i'] == i]
                df_main_main = df_main_main_full[df_main_main_full['i'] ==i] 
                    
                df_predicted_contour = pd.DataFrame()
                df_actual_contour = pd.DataFrame()
                ape_n = []
                mae_n = []
                for n in N[::training_steps]:
                    
                    df_main = df_main_main[df_main_main['n'] == n]
                    df = df_main[df_main['radius'] == R_validation]
                    df_predicted_i_n_filtered = df_predicted_i_filtered[df_predicted_i_filtered['n'] == n]
                    e_actual = pd.Series(df['e'].reset_index(drop=True))
                    e_predicted = pd.Series(df_predicted_i_n_filtered['e'].reset_index(drop=True))
                    
                    fig = plt.figure(figsize=(7, 4))
                    ax = plt.axes()
                    ax.plot(df['y'], e_actual, color = "red")
                    ax.plot(df_predicted_i_n_filtered['y'], e_predicted, color = "blue") 
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
                    
                    error = e_actual - e_predicted
                    percentage_error = (error.abs() / e_actual) * 100
                    ape = percentage_error.mean()
                    ape_n.append(ape)
                    
                    mae = mean_absolute_error(e_actual, e_predicted)
                    mae_n.append(mae)
                    
                    e_predict = df_predicted_i_n_filtered['e'].reset_index(drop=True)
                    df_predicted_contour[n] = e_predict
                    df_actual_contour[n] = e_actual
                
                ape_n_mean = sum(ape_n)/len(ape_n)
                ape_n_mean_i_mean.append(ape_n_mean)
                
                mae_n_mean = sum(mae_n)/len(mae_n)
                mae_n_mean_i_mean.append(mae_n_mean)
            
                
                x1 = np.linspace(0, 50, num=df_predicted_contour.shape[1])
                y = np.linspace(-25, 25, num =df_predicted_contour.shape[0])
                
                colorbarmax = df_predicted_contour.max().max()
                
                X,Y = np.meshgrid(x1,y)
                fig = plt.figure(figsize=(5, 4))
                ax = plt.axes()
                cp=ax.contourf(X,Y,df_predicted_contour, 200, zdir='z', offset=-100, cmap='viridis')
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
                df_predicted_contour.to_csv(file_path+"3dpredictedmultibeam_model_nodes"+str(layer_size)+"_layers"+str(dense_layer)+"_i"+str(i)+".csv")
                df_actual_contour.to_csv(file_path+"3dactualmultibeam_model_nodes"+str(layer_size)+"_layers"+str(dense_layer)+"_i"+str(i)+".csv")
           
            total_ape_mean = sum(ape_n_mean_i_mean)/len(ape_n_mean_i_mean)
            ape_list.append(total_ape_mean)
            total_mae_mean = sum(mae_n_mean_i_mean)/len(mae_n_mean_i_mean)
            mae_list.append(total_mae_mean)
            
            model.save(file_path+"3dmultisinglebeam_model_nodes"+str(layer_size)+"_layers"+str(dense_layer))
    
df_results['nodes'] = nodes
df_results['layers'] = layers1
df_results['ape_list'] = ape_list
df_results['train_test_ape'] = train_test_ape
df_results['mae_list'] = mae_list
df_results['train_test_mae'] = train_test_mae
df_results['time_list'] = time_list
df_results.to_csv(file_path+"optimization_result.csv")
df_training_validation.to_csv(file_path+"training_validation_result.csv")
            
            
#===================================================================            
            
 
