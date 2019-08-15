import sensor
import image
import lcd
import time

clock = time.clock()
lcd.init(freq=15000000)
lcd.direction(0x60)

sensor.reset()
#sensor.OV7725()
sensor.set_vflip(1)
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)
sensor.run(1)
green_threshold   = (0,   80,  -70,   -10,   -0,   30)
red_threshold = (7, 75, 28, 80, 3, 66)
blue_threshold = (0, 68, 6, 73, -104, -21)
yellow_threshold = (60, 97, -27, 25, 30, 74)
orange_threshold = (29, 80, 11, 46, 24, 64)
pink_threshold = (47, 83, 14, 84, -59, 14)
purple_threshold = (5, 87, 6, 32, -43, -14)

while True:
    clock.tick()
    img = sensor.snapshot()
    fps =clock.fps()
    
    # adjest color you want to include here
    blobs = img.find_blobs([red_threshold, green_threshold,blue_threshold, purple_threshold, pink_threshold], merge = True, area_threshold=60, pixel_threshold = 60)
    count = 0
    if blobs:
        count = 0;
        for b in blobs:
            tmp=img.draw_rectangle(b[0:4])
            count += 1;
            #tmp=img.raw_cross(b[5], b[6])
            #c=img.get_pixel(b[5], b[6])
        img.draw_string(280,2, ("%d" %(count)), color=(0,128,0), scale=2)
    img.draw_string(2,2, ("%2.1ffps" %(fps)), color=(0,128,0), scale=2)
    lcd.display(img)
