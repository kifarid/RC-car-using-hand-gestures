# #!/usr/bin/env python  
import RPi.GPIO as GPIO
import time

GPIO.cleanup()
right = 21
left = 20
GPIO.setmode(GPIO.BCM)
GPIO.setup(right, GPIO.OUT,initial=0)
GPIO.setup(left, GPIO.OUT,initial=0)


def stop():

    GPIO.output(right,GPIO.LOW)
    GPIO.output(left,GPIO.LOW)
    print('we are stopping')
    time.sleep(2)

def turn_left():
    
    GPIO.output(left,GPIO.HIGH) 
    GPIO.output(right,GPIO.LOW)
    print('we are righting')
    time.sleep(3)
    stop()
    
def turn_right():
    
    GPIO.output(right,GPIO.HIGH)
    GPIO.output(left,GPIO.LOW)
    print('we are lefting')
    time.sleep(3)
    stop()
    
def forward():
    
    GPIO.output(right,GPIO.HIGH)
    GPIO.output(left,GPIO.HIGH)
    print('we are forwarding')
    time.sleep(3)
    stop()
