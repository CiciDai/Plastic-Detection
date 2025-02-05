import sensor
import image
import lcd
import time
import KPU as kpu

clock = time.clock()
lcd.init(freq=15000000)
#lcd.direction(0x60)

sensor.reset()
sensor.set_vflip(1)
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)
sensor.run(1)
classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
task = kpu.load(0x500000)
anchor = (1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52)
a = kpu.init_yolo2(task, 0.5, 0.3, 5, anchor)
while True:
    clock.tick()
    img = sensor.snapshot()
    code = kpu.run_yolo2(task, img)
    fps =clock.fps()
    img.draw_string(2,2, ("%2.1ffps" %(fps)), color=(0,128,0), scale=2)
    if code:
        for i in code:
            a = img.draw_rectangle(i.rect())
            a = lcd.display(img)
            for i in code:
                lcd.draw_string(i.x(), i.y(), classes[i.classid()], lcd.RED, lcd.WHITE)
                lcd.draw_string(i.x(), i.y()+12, '%1.2f'%i.value(), lcd.RED, lcd.WHITE)
    else:
        a = lcd.display(img)
    #lcd.display(img)
a = kpu.deinit(task)
