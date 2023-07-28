from socket import *
import pandas as pd
from datetime import datetime


HOST = '127.0.0.1'
PORT = 3000
BUFSIZE = 1024
ADDR = (HOST, PORT)

serverSocket = socket(AF_INET, SOCK_STREAM)

serverSocket.bind(ADDR)
print('bind')
filename = 'result_' + datetime.today().strftime("%Y%m%d_%H%M%S")
serverSocket.listen(100)
print('listen')

clientSocekt, addr_info = serverSocket.accept()
print('accept')
print('--client information--')
print(clientSocekt)

delta = []; theta = []; lowAlpha = []; highAlpha = []; lowBeta = []; highBeta = []; lowGamma = []; highGamma = []


temp = 0
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