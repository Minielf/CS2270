import matplotlib.pyplot as plt
import numpy as np

# x = np.arange(0, 61, 1)
# y1 = [-1.5 * (x - 10) * (x - 10) + 150 for x in range(0, 20)]
# y2 = [-1.5 * (x - 30) * (x - 30) + 150 for x in range(20, 40)]
# y3 = [-1.5 * (x - 50) * (x - 50) + 150 for x in range(40, 61)]
#
# target_x = x
# # print(target_x)
# target_y = y1 + y2 + y3
# # print(target_y)
# target = list(zip(target_x, target_y))
# print(target)
#
# plt.plot(target_y)
# plt.title('test pattern for quetch matching algorithm')
# plt.xlabel('input')
# plt.ylabel('function values')
#
# plt.show()
#
# def fx(x):
#     if x < 20:
#         y = -1 * (x - 10) * (x - 10) + 100
#     elif 20 <= x < 40:
#         y = -2 * (x - 30) * (x - 30) + 200
#     else:
#         y = -1 * (x - 50) * (x - 50) + 100
#     return y
#
