'''
LSTM for TN DATA
2018/1/22
by SUDA
'''
import os
import time
import datetime
import numpy
import linecache
import pandas
from math import sqrt
from numpy import concatenate, hstack, vstack
from matplotlib import pyplot
from pandas import read_csv, concat, DataFrame, Series
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential, model_from_json
from keras.layers import Dense, LSTM

# Record Time
str_ymdhms = datetime.datetime.now().strftime(u'%Y/%m/%d %H:%M:%S')
print('Now: %s'% (str_ymdhms))
print('Get started...')

def print_varsize():
    import types
    print("{}{: >15}{}{: >10}{}".format('|','Variable Name','|','  Size','|'))
    print(" -------------------------- ")
    for k, v in globals().items():
        if hasattr(v, 'size') and not k.startswith('_') and not isinstance(v,types.ModuleType):
            print("{}{: >15}{}{: >10}{}".format('|',k,'|',str(v.size),'|'))
        elif hasattr(v, '__len__') and not k.startswith('_') and not isinstance(v,types.ModuleType):
            print("{}{: >15}{}{: >10}{}".format('|',k,'|',str(len(v)),'|'))

def pad_array(val):
    return numpy.array([numpy.insert(pad_col, 0, x) for x in val])
            
# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out): 
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

start1 = time.time()

# load dataset
#dataset_init = read_csv('/home/ubuntu/tmp/DATA/timewidth32/Q0_201706_201708_timewidth32_light2.csv', header=1, index_col=0)
dataset_init = read_csv('/home/ubuntu/tmp/DATA/timewidth32/Q0_201706_201707_timewidth32.csv', index_col=0)
dataset_init = dataset_init.set_index('0')

# Reading Init Parameter
parameter_init = read_csv('temporary_parameter.csv', header=0)
df_para = DataFrame(parameter_init)
df_values = df_para.values

start_ID = df_values[0,1]
end_ID = df_values[1,1]
n_train_hours_init = df_values[2,1]
n_before_hours_init = df_values[3,1]
n_after_hours_init = df_values[4,1]
epoch_num_init = df_values[5,1]
batch_size_vol_init = df_values[6,1]

start_ID = int(start_ID)
end_ID = int(end_ID)
n_before_hours_init = int(n_before_hours_init)
n_after_hours_init = int(n_after_hours_init)
epoch_num_init = int(epoch_num_init)
batch_size_vol_init = int(batch_size_vol_init)

# Make directory
now = datetime.datetime.now()

directory_name1 = '%d_%d_%d_id%d_%d_before1_%d_after1_%d_rate_%.3f_epoch%d_batch%d' % (now.year, now.month, now.day, start_ID, end_ID, n_before_hours_init, n_after_hours_init, n_train_hours_init, epoch_num_init, batch_size_vol_init)

os.makedirs(directory_name1,exist_ok=True)

# Save Init Parameter to CSV

a_values = ['start_ID', 'end_ID', 
            'n_train_hours_init', 
            'n_before_hours_init', 
            'n_after_hours_init',
            'epoch_num_init',
            'batch_size_vol_init']

b_values = [start_ID, 
            end_ID, 
            n_train_hours_init, 
            n_before_hours_init, 
            n_after_hours_init,
            epoch_num_init,
            batch_size_vol_init]

init_array = [a_values, b_values]
init_df = DataFrame(init_array) 
init_df = init_df.T

init_df.to_csv('%s/init_parameter_ID%d_ID%d.csv' % (directory_name1,start_ID,end_ID))


# Initialization for loss & optimizer
loss_name_init = 'mae'
optimizer_name_init = 'adam'  

directory_name2 = '%s/before%d' % (directory_name1,n_before_hours_init)
os.makedirs(directory_name2, exist_ok=True)

directory_name3 = '%s/after%d' % (directory_name2,n_after_hours_init)
os.makedirs(directory_name3, exist_ok=True)
data_frame = DataFrame(index=[], columns=['rmse'])
directory_name4 = '%s/All_id_result%d' % (directory_name3,n_after_hours_init)
os.makedirs(directory_name4, exist_ok=True)

# Beginning from startID to end_ID
for i in range(start_ID,end_ID+1,1):
    os.makedirs(directory_name3, exist_ok=True)
    n_train_hours_init = n_train_hours_init
    n_before_hours = n_before_hours_init
    n_after_hours = n_after_hours_init
    epoch_num = epoch_num_init
    batch_size_vol = batch_size_vol_init

    loss_name = 'mae'
    optimizer_name = 'adam'
    
    str_ymdhms = datetime.datetime.now().strftime(u'%Y/%m/%d %H:%M:%S')
    print('ID%d_THEN: %s'% (i,str_ymdhms))
    print('ID%d start...' % i)
    number_ID = i
    number = i - 1
    
    values = dataset_init.values
    df = DataFrame(values)
    
    values = values[:, number]
    
    values_df = DataFrame(values)
    values_sum = values_df.sum(axis=0)
    a = int(values_sum)
    print(values_sum)

    # ensure all data is float
    values = values.astype('float32')
    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
        
    f = DataFrame(scaled)
    INDEX = len(f.index)
    COLUMNS = len(f.columns)
        
    scaled = scaled.reshape(INDEX, COLUMNS)
        
    # frame as supervised learning
    reframed = series_to_supervised(scaled,
                                    n_before_hours,
                                    n_after_hours)

    # split into train and test sets
    values = reframed.values
    n_train_hours = int(n_train_hours_init * values.shape[0]) +1
    train = values[:n_train_hours, :]
    test = values[n_train_hours:, :]
    # split into input and outputs
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]
    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], train_X.shape[1], 1))
    test_X = test_X.reshape((test_X.shape[0], test_X.shape[1], 1))
    # design network
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss=loss_name, optimizer=optimizer_name)
    # fit network
    history = model.fit(train_X, 
                        train_y, 
                        epochs=epoch_num, 
                        batch_size=batch_size_vol, 
                        validation_data=(test_X, test_y), 
                        verbose=0, 
                        shuffle=False)

    history = DataFrame(history.history)
    history.to_csv('%s/ID_%d.csv' % (directory_name4,i))
    
    # model & hyper-parameter saving
    
    #path1 = '%s/modelsave/modelsave_ID%d.json' %(directory_name1,i)
    #path2 = '%s/hyper_parameter/weights_ID%d.h5' %(directory_name1,i)
    
    #model.to_json(path1)
    #open(os.path.join('%s/modelsave/modelsave_ID%d.json' %(directory_name1,i)), 'w').write(model_json_str)
    #model.save_weights(path2)
    
    #print('save the architecture of a model')
    #json_string = model.to_json()
    #open(os.path.join(f_model,'cnn_model.json'), 'w').write(json_string)
    #yaml_string = model.to_yaml()
    #open(os.path.join(f_model,'cnn_model.yaml'), 'w').write(yaml_string)
    #print('save weights')
    #model.save_weights(os.path.join(f_model,'cnn_model_weights.hdf5'))
 
    # make a prediction
    test_y = test_y.reshape(-1, scaled.shape[1])
    yhat = model.predict(test_X)
    yhat = scaler.inverse_transform(yhat)
    test_y = scaler.inverse_transform(test_y)
    
    # predict
    result = pandas.DataFrame(yhat)
    result.columns = ['predict']
    result['actual'] = test_y
    result.to_csv('%s/predict_ID_%d.csv' % (directory_name4,i))

    # calculate RMSE
    rmse = sqrt(mean_squared_error(test_y, yhat))
    series = Series([rmse], index=data_frame.columns)
    data_frame = data_frame.append(series,ignore_index = True)
    
    ### save weights
    json_string = model.to_json()
    open('%s/lstm_model_ID%d.json' % (directory_name1,i), 'w').write(json_string)
    model.save_weights('%s/lstm_weights_ID%d.h5' %(directory_name1,i))

    del train
    del test
    del values
    del values_df
    del scaler
    del scaled
    del reframed
    del df
    del history
    del test_X
    del test_y
    del train_X
    del train_y
    del f
    del yhat
    #del path1
    #del path2
            
# Making Data for RMSE statistics 
data_frame.to_csv('%s/after_RMSE.csv' % (directory_name3))

data = read_csv('%s/after_RMSE.csv' % (directory_name3), header=0, index_col=0)
data = DataFrame(data)

s = data.sum()
ave = data.mean()
median = data.median()
var = data.var()
std = data.std()
skew = data.skew()
kurt = data.kurt()

frame_name = ['sum','ave','median','var','std','skew','kurt']
frame_name = DataFrame(frame_name)

s = s.values
ave = ave.values
median = median.values
var = var.values
std = std.values
skew = skew.values
kurt = kurt.values

data_numpy = [s,ave,median,var,std,skew,kurt]
data = DataFrame(data_numpy)

# Save RMSE statistics
data = pandas.concat([frame_name, data], axis=1)
data.to_csv('%s/after_RMSE_statistics.csv' % (directory_name3))

print('after%d...done...' % (n_after_hours_init))
print('before%d...done...' % (n_before_hours_init))

# Output Calculation Time
str_ymdhms = datetime.datetime.now().strftime(u'%Y/%m/%d %H:%M:%S')
print('THEN: %s'% (str_ymdhms))
elapsed_time = time.time() - start1

time_index = ['time']
time_index = DataFrame(time_index)
elapsed_time = [elapsed_time]
elapsed_time = DataFrame(elapsed_time)

time = pandas.concat([time_index, elapsed_time], axis=1)

time.to_csv('%s/cul_time.csv' % (directory_name3))

print('done!!!')