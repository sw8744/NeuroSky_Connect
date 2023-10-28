import serial
import time

while True:
    signal = input()
    ser = serial.Serial('COM13', 9600)
    time.sleep(2)
    print(ser.writable())
    signal = signal.encode()
    # print(signal)
    ser.write(signal)
    ser.close()