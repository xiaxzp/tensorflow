import tensorflow as tf
import numpy as np
import pandas as pd
from pandas_datareader import data as pdr
import datetime
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler as mms
import matplotlib.pyplot as plt

yf.pdr_override()
start_date="2016-1-1"
end_date="2018-1-1"
now = datetime.datetime(2020,1,1).timestamp()
sc = mms(feature_range=(0,1))
def normalize_windows(win_data):
  """ Normalize a window
  Input: Window Data
  Output: Normalized Window

  Note: Run from load_data()

  Note: Normalization data using n_i = (p_i / p_0) - 1,
  denormalization using p_i = p_0(n_i + 1)
  """
  # norm_data = []
  # for w in win_data:
  #     norm_win = [(float(p) / float(w[0])) for p in w]
  #     # norm_win = [[(float(p) / float(now)), 0b00000001 / 256] for p in w]
  #     # norm_win = [[float(p), 0b00000001 ] for p in w]
  #     # norm_win = [float((p << 4) + 0b0001) for p in w]
  #     norm_data.append(norm_win)
  return win_data

gafataDict={"谷歌":"GOOG","亚马逊":"AMZN","Facebook":"FB","苹果":"AAPL","阿里巴巴":"BABA","腾讯":"0700.hk"}
googDF=pdr.get_data_yahoo(gafataDict["谷歌"],start_date,end_date)
DJIDF=pdr.get_data_yahoo("DJI",start_date,end_date)
# amznDF=pdr.get_data_yahoo(gafataDict["亚马逊"],start_date,end_date)
# fbDF=pdr.get_data_yahoo(gafataDict["Facebook"],start_date,end_date)
# aaplDF=pdr.get_data_yahoo(gafataDict["苹果"],start_date,end_date)
# babaDF=pdr.get_data_yahoo(gafataDict["阿里巴巴"],start_date,end_date)
# txDF=pdr.get_data_yahoo(gafataDict["腾讯"],start_date,end_date)
# googDFY = [ (googDF['Close'][i] - googDF['Open'][i] ) for i in range(len(googDF['Close']) - 1)]
# print(googDF, DJIDF)
googDFY = [ [googDF['Close'][i], DJIDF['Close'][i]] for i in range(len(googDF['Close']) - 1)]
googDFTime = [ int(x.timestamp()) for x in googDF.index.to_pydatetime()]
seq_len = 5
inp = []
out = []
googDFY = np.array(googDFY)
googDFY = googDFY.reshape(-1, 2)
dataDFY = sc.fit_transform(googDFY)

for i in range(len(dataDFY) - seq_len - 1):
  inp.append(dataDFY[i: i + seq_len])
  out.append([dataDFY[i + seq_len][0]])
  # out.append([x[0] for x in dataDFY[i + seq_len: i + seq_len + seq_len]])
inp = normalize_windows(inp)
inp = np.array(inp)
out = np.array(out)

x_tr = inp[:370]
y_tr = out[:370]
x_te = inp[370:]
y_te = out[370:]
x_tr = x_tr.reshape(x_tr.shape[0],x_tr.shape[1],-1)
# y_tr = y_tr.reshape(-1, seq_len)
x_te = x_te.reshape(x_te.shape[0],x_te.shape[1],-1)
# y_te = y_te.reshape(-1, seq_len)
print(x_tr.shape)
# x_te = x_te.reshape(-1, x_te.shape[0],x_te.shape[1],x_te.shape[2])

# xarray = (googDF['Date'][:200] + np.ones(200) * 0b00000001 + np.ones(200) * 1 ).reshape([-1, 1,3], order='F')
# yarray = (googDF['Close'][:200])



model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(units = 5, dropout=0.1, return_sequences=True),
    tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'),
    tf.keras.layers.MaxPool1D(pool_size=2),
    tf.keras.layers.LSTM(units = 20, dropout=0.1),
    # tf.keras.layers.LSTM(40, dropout=0.2, input_shape=(x_tr.shape[1],x_tr.shape[2]), return_sequences=True),
    # tf.keras.layers.LSTM(40, dropout=0.2, input_shape=(x_tr.shape[1],x_tr.shape[2])),
    tf.keras.layers.Dense(units = 1)
])

model.compile(optimizer='adam',
              loss='mse')

model.fit(x_tr, y_tr, batch_size=5, epochs=100)
# # model.evaluate(xarraytest, yarraytest)
output = []
init = x_te[10]
print(init)
for i in range(seq_len):
  tparr = init.reshape(1,seq_len,2)
  tp = model.predict(tparr, batch_size=1)
  print(tp, y_te[i])
  init = init[1:]
  init = np.append(init, [[tp[0][-1], x_te[10][0][1]]], 0)
  print('change', init)
  output.append(tp[0][-1])

outputp = model.predict(x_te, batch_size=1)
outputpt = model.predict(x_tr, batch_size=4)
print(outputp, y_te)
# plt.figure(figsize=(8,5))
calc = 0
calc1 = 0
calc2 = 0
for i in range(len(outputpt)):
  # if i > 0 and (outputpt[i][0][0] - outputpt[i-1][0][0]) * (y_tr[i][0] - y_tr[i-1][0]) > 0:
  #   calc = calc + 1
  # if i > 0 and (outputpt[i][1][0] - outputpt[i-1][1][0]) * (y_tr[i][1] - y_tr[i-1][1]) > 0:
  #   calc1 = calc1 + 1
  # if i > 0 and (outputpt[i][2][0] - outputpt[i-1][2][0]) * (y_tr[i][2] - y_tr[i-1][2]) > 0:
  #   calc2 = calc2 + 1
  if i > 0 and (outputpt[i][0] - outputpt[i-1][0]) * (y_tr[i][0] - y_tr[i-1][0]) > 0:
    calc = calc + 1

print(float(calc) / 360,float(calc1) / 360,float(calc2) / 360)


plt.subplot(3,  1,  1)
data_list=[str(i) for i in range(0, seq_len)]
data_list2=[str(i) for i in range(0, 370)]
plt.plot(data_list[0:seq_len], np.squeeze(y_te[10:10+seq_len]), color='r', linewidth=2)
plt.plot(data_list[0:seq_len], np.squeeze(outputp[10:10+seq_len]), color='b', linewidth=2)
# plt.plot(data_list[0:seq_len], np.squeeze([x[0] for x in outputp[0]]), color='b', linewidth=2)
plt.subplot(3,  1,  2)
plt.plot(data_list[0:seq_len], np.squeeze(y_te[10: 10+seq_len]), color='r', linewidth=2)
plt.plot(data_list[0:seq_len], np.squeeze(output[0: seq_len]), color='b', linewidth=2)
# plt.plot(data_list[0:seq_len], np.squeeze([x[0] for x in outputp[10]]), color='b', linewidth=2)
plt.subplot(3,  1,  3)
plt.plot(data_list2, np.squeeze(y_tr), color='r', linewidth=2)
plt.plot(data_list2, np.squeeze(outputpt), color='b', linewidth=2)
# plt.plot(data_list2, np.squeeze([y[2] for y in y_tr]), color='r', linewidth=2)
# plt.plot(data_list2, np.squeeze([x[2][0] for x in outputpt]), color='b', linewidth=2)
# plt.plot(data_list[0:seq_len], np.squeeze(output), color='g', linewidth=2)
# plt.plot(data_list[0:seq_len], np.squeeze(y_te)[0:seq_len], color='r', linewidth=2)
# plt.plot(data_list[0:seq_len], np.squeeze(output)[0:seq_len], color='g', linewidth=2)
# plt.plot(data_list[0:seq_len], np.squeeze(outputp)[0:seq_len], color='b', linewidth=2)
# data_list=[str(i) for i in range(0, 340)]
# plt.plot(data_list[0:340], np.squeeze(y_tr)[0:340], color='r', linewidth=2)
# plt.plot(data_list[0:340], np.squeeze(outputpt)[0:340], color='b', linewidth=2)
plt.xlabel('time')
plt.ylabel('value')
#plt.plot(list(range(len(test_predict))), test_predict, color='b')
#plt.plot(list(range(len(test_y))), test_y, color='r')
plt.show()

