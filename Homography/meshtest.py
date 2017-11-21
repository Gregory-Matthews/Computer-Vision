import sys
import numpy as np


# (Width * Height) 3 x 2
p1 = [10, 11, 12]
p2 = [20, 21, 22]
p3 = [30, 31, 32]
p4 = [40, 41, 42]
p5 = [50, 51, 52]
p6 = [60, 61, 62]
p7 = [70, 71, 72]
p8 = [80, 81, 82]
p9 = [90, 91, 92]


image = np.array([[p1, p2, p3], [p4, p5, p6]])

shape = image.shape
y = shape[0]
x =  shape[1]


row, col = np.mgrid[0:x, 0:y]
mesh = np.mgrid[0:x, 0:y]
#print image.shape

print image[row[1][1]][col[1][1]]

print row[1][1]
print ""
print col

for x, y in zip(row, col):
    for x2, y2 in zip(x, y):
        print "%d %d" % (x2, y2)



#print image[row[0][0]][col[0][0]]
#print image[row[1][2]][col[0][0]]


#print mesh