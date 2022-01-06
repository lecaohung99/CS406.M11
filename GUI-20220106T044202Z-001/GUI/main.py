import os
import shutil

count = 0
f = open("1.txt")
for i in f.readlines():
    original = r"D:\Train\masks\{}.png".format(i.rstrip())
    target = r"D:\Test2\masks\{}.png".format(i.rstrip())
    shutil.move(original, target)
    count += 1
    print(count, "D:\Train\images\{}.png".format(i.rstrip()))