#!/usr/bin/python3
# coding: utf-8
# solver.py

import csv
import numpy

with open('experiment.txt', 'r') as csvfile:
    # parse
    reader = csv.DictReader(csvfile, dialect='excel-tab')
    headers = reader.fieldnames
    print(headers)

    # # y_t = sum_t(b*x_t) - regression model, where y_t = line_vars[0], x_t = line_vars[1:]
    line_vars = [[] for i in range(len(headers))]

    for row in reader:
        for index in range(len(headers)):
            line_vars[index].append(row[headers[index]])

    for index in range(len(headers)):
        line_vars[index] = [float(i) for i in line_vars[index]]

    # fill in
    Y = numpy.array(line_vars[0])
    XT = numpy.array([line_vars[i + 1] for i in range(len(headers) - 1)])

    X = XT.transpose()
    YT = Y[numpy.newaxis, :].T

    print(X)
    print(YT)

    # compute
    A = XT @ X
    B = XT @ Y

    # # Ax = B, where A = (X^T*X), x = b - our goal, B = X^T*Y
    x = numpy.linalg.solve(A, B)

    print(x)
