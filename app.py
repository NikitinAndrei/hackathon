# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 11:40:37 2021

@author: Иван
"""
import tkinter as tk
import tkinter.filedialog as fd
import tkinter.messagebox as mb
from time import sleep
import tkinter as tk
from PIL import Image, ImageTk
from itertools import count

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        btn_file = tk.Button(self, text="Выбрать файл",
                             command=self.choose_file)
      
        btn_file.pack(padx=500, pady=100)
        

    def choose_file(self):
        filetypes = (
                     ("Данные", "*.xlsx"),
                     )
        filename = fd.askopenfilename(title="Открыть файл", initialdir="/",
                                      filetypes=filetypes)
        if filename:
            print(filename)
            app.destroy()
class App2(tk.Tk):
    def __init__(self):
        super().__init__()
       
        self.btn_save = tk.Button(self, text="Сохранить", command=self.save_file)

        self.btn_save.pack(padx=500, pady=100)

    def save_file(self):
        contents = "ааа"
        new_file = fd.asksaveasfile(title="Сохранить файл", defaultextension=".xls",
                                    filetypes=(("Результат", "*.xlsx"),))
        if new_file:
            new_file.write(contents)
            new_file.close() 
            app.destroy()
class ImageLabel(tk.Label):
    """a label that displays images, and plays them if they are gifs"""
    def load(self, im):
        if isinstance(im, str):
            im = Image.open(im)
        self.loc = 0
        self.frames = []
        try:
            for i in count(1):
                self.frames.append(ImageTk.PhotoImage(im.copy()))
                im.seek(i)
        except EOFError:
            pass
        try:
            self.delay = im.info['duration']
        except:
            self.delay = 100
        if len(self.frames) == 1:
            self.config(image=self.frames[0])
        else:
            self.next_frame()
    def unload(self):
        self.config(image="")
        self.frames = None
    def next_frame(self):
        if self.frames:
            self.loc += 1
            self.loc %= len(self.frames)
            self.config(image=self.frames[self.loc])
            self.after(self.delay, self.next_frame)

if __name__ == "__main__":
    
    app = App()
    app.mainloop()
    app = ImageLabel(tk.Tk())
    app.pack()
    app.load('E:\RSV_hacaton\cat.gif')
    app.after(3000,lambda:app.destroy())
    app.mainloop()
    app = App2()
    app.mainloop()
    app.close()
    
"""
import pandas as pd
import numpy as np
from statistics import mean
data_train = '/content/Train.xlsx'
def load_train(data):
    '''
    data: Ссылка на тренировочный датасет
    '''
    data_train = pd.read_excel(data)
    data_train = data_train[1:data_train.size]
    numpy_dataset = []
    for j in range(1,len(data_train.columns)):
        numpy_dataset += [np.array(data_train[data_train.columns[j]])]
    for i in range(len(numpy_dataset)):
        for j in range(len(numpy_dataset[i])):
            if np.isnan(numpy_dataset[i][j]):
                numpy_dataset[i][j] = 0
    return numpy_dataset
data_train = load_train(data_train)
def load_test(file):
    data = pd.read_excel(file)
    data_copy = data.copy()
    data_copy = data_copy.iloc[:, 1:]
    numpy_data = np.array(data_copy)
    
    for i in range(1, numpy_data.shape[1]):
        if(i == 1):
            Number_of_Forecast = np.count_nonzero(numpy_data[:, i] == 'Forecast')
        else:
            Number_of_Forecast = np.append(Number_of_Forecast, np.count_nonzero(numpy_data[:, i] == 'Forecast'))
    
    out = []
    for i in range(0, numpy_data.shape[1], 1):
        out.append(numpy_data[:, i][numpy_data[:, i] != 'Forecast'])
    
    if(len(set(Number_of_Forecast)) == 1):
        return out, "Безусловный прогноз"
    elif(np.count_nonzero(Number_of_Forecast == 0)):
        return out, "Условный прогноз"
    elif(any((i > mean(Number_of_Forecast)) for i in Number_of_Forecast)):
        return out, "Прогноз с задержкой статистики"
    else:
        return "error"
    
data = pd.read_excel('/content/Test_example2.xlsx')
data_copy = data.copy()

dates = np.array(data_copy.tail(6)['Unnamed: 0'])
number_dates = len(dates)
dates = dates.reshape(number_dates, 1)

header = np.array(data_copy.columns)
number_header = len(header)
header = header.reshape(1, number_header)

data_copy = data_copy.iloc[:, 1:]

numpy_data = np.array(data_copy)
numpy_predict = numpy_data[2:8, :]
numpy_output = np.hstack((dates, numpy_predict))
numpy_output = np.vstack((header, numpy_output))

output_dataset = pd.DataFrame(numpy_output)
output_dataset.to_excel('/content/aboba.xlsx', index=False)
test = pd.read_excel('/content/aboba.xlsx')
test_copy = test.copy()
g = load_train('/content/Train.xlsx')
n = np.array(g)
b = []
for i in range(69):
  val = g[i][30:147]
  a =list(val)
  b.append(a)
c = []

for i in b:
  c.append(np.array(i))
for i in range(len(c)):
  c[i] = c[i].reshape((len(c[i]), 1))
from numpy import array
from numpy import hstack
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed

# split a multivariate sequence into samples
def split_sequences(sequences, n_steps_in, n_steps_out):
	x, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out
		# check if we are beyond the dataset
		if out_end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix:out_end_ix, :]
		x.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)



# horizontally stack columns
dataset = hstack(c)
# choose a number of time steps
n_steps_in, n_steps_out = 10, 5
# covert into input/output
X, y = split_sequences(dataset, n_steps_in, n_steps_out)
# the dataset knows the number of features, e.g. 2
n_features = X.shape[2]
model = Sequential()
model.add(LSTM(200, activation='relu', input_shape=(n_steps_in, n_features)))
model.add(RepeatVector(n_steps_out))
model.add(LSTM(200, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(n_features)))
model.compile(optimizer='adam', loss='mse')
from math import exp
from statistics import variance as var
from random import random
data_1 = [[j for j in range(24)] for i in range(5)]
data_2 = [[j + 1 + random() for j in range(24)] for i in range(5)]

def scoring(real_data, predicted_data, n):
    '''
    real_data: фактические значения
    predicted_data: предсказанные
    n: сколько предсказывали
    '''
    x = [real_data[i][-n:] for i in range(len(real_data))]
    x_hat = [predicted_data[i][-n:] for i in range(len(predicted_data))]
    M = len(real_data)
    # M = real_data.shape[0]
    alpha = 12
    # Количество точек для прогнозирования
    Km = [sum(x[i]) for i in range(len(x))]

    Nm = len(real_data)
    # Горизонт прогнозирования
    Hm = n
    
    Dm = [var([predicted_data[j][i] - predicted_data[j][i-12] for i in range(len(predicted_data[j]) - n, len(predicted_data[j]))]) for j in range(len(predicted_data))]

    WMSFEm = [1 / Km[k] * (sum([sum([(x[i][h] - x_hat[i][h])**2 / (h*Dm[i]) for h in range(1,Hm)]) for i in range(Nm)])) for k in range(len(Km))]
    Score = 1 / M * sum(WMSFEm)
    return Score
"""