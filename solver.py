#!/usr/bin/python3
# coding: utf-8
# solver.py

import csv
import numpy

fieldnames = ['y', 'x1', 'x2', 'x3']

with open('experiment.txt', 'r') as csvfile:
    reader = csv.DictReader(csvfile, fieldnames, dialect='excel-tab')
    array = list(reader)

# y_t = sum_t(b*x_t)
line_y = []
line_x1 = []
line_x2 = []
line_x3 = []

for row in array:
    line_y.append(row['y'])
    line_x1.append(row['x1'])
    line_x2.append(row['x2'])
    line_x3.append(row['x3'])

line_y = [float(i) for i in line_y]
line_x1 = [float(i) for i in line_x1]
line_x2 = [float(i) for i in line_x2]
line_x3 = [float(i) for i in line_x3]

Y = numpy.array(line_y)
XT = numpy.array([line_x1, line_x2, line_x3])  # transposed? transposed.
print(XT)
print(Y)

# Ax = B, where A = (X^T*X), x = b, B = X^T*Y
X = XT.transpose()
YT = Y[numpy.newaxis, :].T  # or .transpose() - no matter there is, visual only
print(X)
print(YT)

A = XT @ X
B = XT @ Y

x = numpy.linalg.solve(A, B)
print(x)


