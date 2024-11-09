import time
import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BCM)  # Use Broadcom pin numbering
sound_pin = 17  # GPIO 17 Depending on the sensors pos connection
GPIO.setup(sound_pin, GPIO.IN)

while True:
    if GPIO.input(sound_pin):
        print("Sound detected!")
    else:
        print("No sound detected")
    time.sleep(0.5)
