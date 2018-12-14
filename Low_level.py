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


##while(True):
 #   turn_left()
  #  turn_right()
   # stop()
    #forward()
#    break
#GPIO.cleanup()


   
#right = 20
#left = 21
#GPIO.setmode(GPIO.BOARD)
 
# PIN 7 AND 3.3V
# normally 0 when connected 1
#GPIO.setup(cnl, GPIO.IN, GPIO.PUD_DOWN)
#try:
#while(True):
#print(GPIO.input(cnl))
#time.sleep(1)
#except KeyboardInterrupt:
#GPIO.cleanup()
#print(“Exiting”)


#led = 8
#set numbering mode for the program 
#GPIO.setmode(GPIO.BOARD)
#setup led(pin 8) as output pin
#GPIO.setup(led, GPIO.OUT,initial=0)
#try:
#turn on and off the led in intervals of 1 second
#while(True):
#turn on, set as HIGH or 1
#GPIO.output(led,GPIO.HIGH)
#print(“ON”)
#time.sleep(1)
#turn off, set as LOW or 0
#GPIO.output(led, GPIO.LOW)
#print(“OFF”)
#time.sleep(1)
#except KeyboardInterrupt:
#cleanup GPIO settings before exiting
#GPIO.cleanup()
#print(“Exiting...”)
 