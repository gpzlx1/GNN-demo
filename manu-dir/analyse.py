import os
import sys
import csv
from numpy.core.numeric import convolve

from numpy.lib.function_base import average, iterable

PATH = sys.argv[1]

csv_data = []
kernel_name = {}
with open(PATH, 'r') as f:
    csv_reader = csv.reader(f)
    for step, data in enumerate(csv_reader):
        csv_data.append(data)
        if step == 0:
            continue
        name = data[-1]
        if name in kernel_name:
            kernel_name[name] = kernel_name[name] + 1
        else:
            kernel_name[name] = 1

print(kernel_name.values())
one_iteration_count = 118

start_end_kernel = None
for key, value in kernel_name.items():
    if value == 118:
        start_end_kernel = key
        break

kernel_name_lists = []
iteration_begin = []
for step, data in enumerate(csv_data):
    kernel_name = data[-1]

    if start_end_kernel == kernel_name:
        iteration_begin.append(step)


print(iteration_begin)
csv_data = csv_data[iteration_begin[0]:iteration_begin[-1]]

kernel_lists = []
timeline_lists = []

iterations_count = len(iteration_begin) -1
for i in range(iterations_count):
    begin = iteration_begin[i]
    end = iteration_begin[i + 1]
    kernels = []
    timeline = []
    for data in csv_data[begin:end]:
        time = int(data[1])
        name = data[-1]
        kernels.append(name)
        timeline.append(time)

    kernel_lists.append(kernels)
    timeline_lists.append(timeline)

timeline_sum = []
for step, time in enumerate(timeline_lists):
    print(sum(time))
    timeline_sum.append(sum(time))

print(average(timeline_sum[10:30]))

with open("one_iterations.csv", 'w') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(["name", "time"])
    print(len(kernel_lists[15]))
    for data in zip(kernel_lists[15], timeline_lists[15]):
        csv_writer.writerow(data)