from socket import *
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import serial
def load_data(idlePath:str,sleepPath:str,sleepyPath:str):
    df1 = pd.read_csv(idlePath).assign(result=lambda x:'IDLE')
    df2 = pd.read_csv(sleepPath).assign(result=lambda x:'SLEEP')
    df3 = pd.read_csv(sleepyPath).assign(result=lambda x:'SLEEPY')

    result = pd.concat([df1,df2,df3])
    return result

df = load_data('result_idle.csv','result_sleep.csv','result_sleepy.csv') # 데이터 불러오는 코드임.

df

Y = df['result'].to_numpy()
X = df.drop(columns=['result']).to_numpy()

rescale = StandardScaler()
rescale.fit(X)
rescale_X = rescale.transform(X)

x_train, x_test, y_train, y_test = train_test_split(rescale_X,Y,test_size=0.2) # 훈련용 데이터 전체의 80% 비중, 테스트 데이터 전체의 비중 20%

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(y_train)
y_train = le.transform(y_train)
y_test = le.transform(y_test)

from lightgbm import LGBMClassifier
print("LightGBM Model Training Start")
model4 = LGBMClassifier(n_estimators=200)
model4.fit(x_train,y_train)
print("Train Score :", model4.score(x_train,y_train), "Test Score :", model4.score(x_test,y_test))

unique, counts = np.unique(y_test, return_counts=True)

le.classes_

unique,counts = np.unique(Y, return_counts=True)

print('Training Completed')
print('----------------------------------------------')
resize_test = rescale.transform(X)

bool_eeg = input('Do you want to measure eeg signal? (y/n) ')
if bool_eeg == 'y':
    bool_eeg = True
elif bool_eeg == 'n':
    bool_eeg = False
bool_arduino = input('Do you want to connect with Arduino? (y/n) ')
if bool_arduino == 'y':
    bool_arduino = True
elif bool_arduino == 'n':
    bool_arduino = False


if bool_eeg:
    HOST = '127.0.0.1'
    PORT = 3000
    BUFSIZE = 1024
    ADDR = (HOST, PORT)

    serverSocket = socket(AF_INET, SOCK_STREAM)

    serverSocket.bind(ADDR)
    filename = 'result_' + datetime.today().strftime("%Y%m%d_%H%M%S")
    serverSocket.listen(100)
    print('Waiting for Connection...')

    clientSocekt, addr_info = serverSocket.accept()
    print('Connection Accepted')
    print('--Client Information--')
    print(clientSocekt)

    delta = []; theta = []; lowAlpha = []; highAlpha = []; lowBeta = []; highBeta = []; lowGamma = []; highGamma = []
    if bool_arduino:
        serPort = input('Enter the Serial port Number (ex. COM5')
        ser = serial.Serial(serPort, 9600)
    temp = 0
    times = 0
    result_10s = []
    while True:
        try:
            data = clientSocekt.recv(65535)
            string = data.decode()
            if temp == 0:
                temp += 1
                continue
            dict_string = eval(string)
            if len(dict_string) == 1:
                continue
            if dict_string['poorSignalLevel'] != 0:
                continue

            print(dict_string)
            delta.append(dict_string['eegPower']['delta'])
            theta.append(dict_string['eegPower']['theta'])
            lowAlpha.append(dict_string['eegPower']['lowAlpha'])
            highAlpha.append(dict_string['eegPower']['highAlpha'])
            lowBeta.append(dict_string['eegPower']['lowBeta'])
            highBeta.append(dict_string['eegPower']['highBeta'])
            lowGamma.append(dict_string['eegPower']['lowGamma'])
            highGamma.append(dict_string['eegPower']['highGamma'])
            df = pd.DataFrame(delta, columns=['delta'])
            df['theta'] = theta
            df['lowAlpha'] = lowAlpha
            df['highAlpha'] = highAlpha
            df['lowBeta'] = lowBeta
            df['highBeta'] = highBeta
            df['lowGamma'] = lowGamma
            df['highGamma'] = highGamma
            df.to_csv(filename + '.csv', index=False)
            testData = pd.read_csv(filename + '.csv').to_numpy()
            testData = rescale.transform(testData)
            print('LightGBM', le.inverse_transform(model4.predict(testData))[-1])
            if times < 10:
                result_10s.append(le.inverse_transform(model4.predict(testData))[-1])
                times += 1
            else:
                num_idle = 0;
                num_sleepy = 0;
                num_sleep = 0
                for i in result_10s:
                    if i == "IDLE":
                        num_idle += 1
                    elif i == "SLEEPY":
                        num_sleepy += 1
                    elif i == "SLEEP":
                        num_sleep += 1
                num_array = [num_idle, num_sleepy, num_sleep]
                num_max = max(num_idle, num_sleepy, num_sleep)
                num_max_idx = str(num_array.index(num_max))
                print('----------------------------------------------')
                print('MAX_Condition', num_max_idx)
                if bool_arduino:
                    ser.write(str.encode(num_max_idx))
                times = 0
                result_10s = []
        except IOError:
            print('----------------------------------------------')
            print('End Listening')
            print('delta', delta)
            print('theta', theta)
            print('lowAlpha', lowAlpha)
            print('highAlpha', highAlpha)
            print('lowBeta', lowBeta)
            print('highBeta', highBeta)
            print('lowGamma', lowGamma)
            print('highGamma', highGamma)
            break
        except SyntaxError:
            continue
else:
    try:
        fname = input('Enter the file name (ex. abc.csv) : ')
        testData = pd.read_csv(fname).to_numpy()
        testData = rescale.transform(testData)
        print('LightGBM', le.inverse_transform(model4.predict(testData)))
        num_idle = 0;
        num_sleepy = 0;
        num_sleep = 0
        for i in le.inverse_transform(model4.predict(testData)):
            if i == "IDLE":
                num_idle += 1
            elif i == "SLEEPY":
                num_sleepy += 1
            elif i == "SLEEP":
                num_sleep += 1
        num_array = [num_idle, num_sleepy, num_sleep]
        num_max = max(num_idle, num_sleepy, num_sleep)
        num_max_idx = str(num_array.index(num_max))
        print('----------------------------------------------')
        print('MAX_Condition', num_max_idx)
        if bool_arduino:
            serPort = input('Enter the Serial port Number (ex. COM5')
            ser = serial.Serial(serPort, 9600)
            if ser.readable():
                ser.write(str.encode(num_max_idx))
    except FileNotFoundError:
        print('File Not Found')
