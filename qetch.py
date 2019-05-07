import numpy as np
import matplotlib.pyplot as plt
from head_and_shoulder import *
from qetch_test1 import *
from qetch_test2 import *
from qetch_test3 import *
from load_data import *
import math
import random

# class declaration
class Point:
    def __init__(self, x, y, ema):
        self.x = x
        self.y = y
        self.ema = ema

class Segment:
    def __init__(self, points):
        self.points = points



# HELPER FUNCTIONS
# convert a continuous input to discrete
def con_to_dis(end, step_size):
    x = np.arange(0, end, step_size)
    y = [fx(i) for i in x]
    return list(zip(x, y))

def fx(x):
    y = math.cos(x)
    # if x < 20:
    #     y = -1.5 * (x - 10) * (x - 10) + 150
    # elif 20 <= x < 40:
    #     y = -1.5 * (x - 30) * (x - 30) + 150
    # else:
    #     y = -1.5 * (x - 50) * (x - 50) + 150
    return y

# get back all the points from segments
def desegmentation(segs):
    P = []
    for seg in segs:
        P += seg
    return P

# based on the segments from candidates, get the corresponding target segments
def gen_Q_segs(C_segs):
    Q_segs = []
    for i in range(len(C_segs)):
        temp = []
        for p in C_segs[i]:
            temp.append(Point(p.x, fx(p.x), 0))  # hard code the query input
        Q_segs.append(temp)
    return Q_segs

# format the input data to Point type
def format_points(raw_data, smoothed_data):
    points = []
    for i in range(len(str_days)):
        points.append(Point(i, raw_data.loc[str_days[i], 'NFLX'], smoothed_data.loc[str_days[i], 'NFLX']))
    return points

# format the target data to Point type
def format_target_points(inputs):
    points = []
    for i, j in inputs:
        points.append(Point(i, j, j))
    return points

# divide segs to a list of candidates, each with k segments
# input: a list of list, segments
# output: a list of list, [segs[0],segs[1], ...,segs[k-1]], [segs[1],segs[2], ...segs[k]],
def gen_candidates(segs, k):
    cand_list = []

    for i in range(len(segs)-k+1):
        cand_list.append(segs[i:i+k])
    return cand_list

def total_err(errs):
    sm = 0
    for err in errs:
        sm += abs(err)
    return sm



# QETCH ALGORITHMS

# divide a list of points P into different segments
# elements in each segment are continuous elements with same monotonicity
# return a list of list
def segmentation(P):
    res = []
    segment = []
    mono = 1 if P[1].y > P[0].y else -1
    for i in range (1, len(P)):
        temp_mono = 1 if P[i].y > P[i-1].y else -1
        # start a new segment when monotonicity changes
        if temp_mono != mono:
            res.append(segment)
            segment = []
            mono = temp_mono
        segment.append(P[i])
    res.append(segment)
    return res

# C, Q are a series of Points: C[0], C[1]...
def scaling(C, Q):
    Gx = (C[-1].x - C[0].x) / (Q[-1].x - Q[0].x)
    Gy = (max(c.y for c in C) - min(c.y for c in C)) / (max(q.y for q in Q) - min(q.y for q in Q))
    return Gx, Gy

def distortionErr(C, Q):
    Gx, Gy = scaling(C, Q)
    C_segs = segmentation(C)
    Q_segs = gen_Q_segs(C_segs)

    Rx, Ry, LDE = [], [], []

    # calculate Rx, Ry for each segment
    for i in range(len(C_segs)):
        Rx.append((C_segs[i][-1].x - C_segs[i][0].x) / (Gx * (Q_segs[i][-1].x - Q_segs[i][0].x)))
        Ry.append((max(c.y for c in C_segs[i]) - min(c.y for c in C_segs[i])) / (Gy * (max(q.y for q in Q_segs[i]) - min(q.y for q in Q_segs[i]))))
        LDE.append(math.log(Rx[i] * Rx[i]) + math.log(Ry[i] * Ry[i]))
    return Rx, Ry, LDE

def shapeErr(C, Q):
    Gx, Gy = scaling(C, Q)
    Rx, Ry, LDE = distortionErr(C, Q)
    C_segs = segmentation(C)
    Q_segs = gen_Q_segs(C_segs)
    height_C = max(c.y for c in C) - min(c.y for c in C)
    SE = []

    for i in range(len(C_segs)):
        temp = 0
        for j in range(len(C_segs[i])):
            temp += (Gy * Ry[i] * Q_segs[i][j].y - C_segs[i][j].y) / height_C
        SE.append(1 / len(C_segs[i]) * temp)

    Dist = []
    for i in range(len(C_segs)):
        Dist.append(LDE[i] + SE[i])
    return Dist

# noise: a segment with height < 1% overall height
# merge noise with adjacent segments
def remove_noise(segs):
    clean_segs = []
    max_h = 0
    # calcute overall height
    for seg in segs:
        h = max(p.y for p in seg) - min(p.y for p in seg)
        max_h = max(h, max_h)

    # detect noise and merge with next segment
    noise = []
    for seg in segs:
        h = max(p.y for p in seg) - min(p.y for p in seg)
        if h < 0.2 * max_h:
            noise += seg
        else:
            merged_seg = []
            if len(noise) != 0:
                merged_seg = noise + seg
            else:
                merged_seg = seg
            clean_segs.append(merged_seg)
            noise = []
    # merge trailing noise with last segment
    if noise:
        clean_segs[-1] += noise

    return clean_segs


# ANOMALY DETECTION ALGORITHM
#return average distance from a point to its smoothed point
def calc_delta(cand):
    dist, cnt = 0, 0
    for seg in cand:
        dist += sum([((p.y - p.ema) * (p.y - p.ema)) for p in seg])
        cnt  += len(seg)
    return dist/cnt


# exponential moving average
short_ema = msft.ewm(span=8, adjust=False).mean()

# converging not used. It may lead to over-smoothing
# while abs(len((segmentation(format_points(short_ema)))) - len((segmentation(format_points(msft))))) / len((segmentation(format_points(msft)))) >= 0.1:
#     msft = short_ema
#     short_ema = short_ema.ewm(span=20, adjust=False).mean()



target = con_to_dis(2 * math.pi, 0.2)
points = format_points(short_ema, msft)
segs = remove_noise(segmentation(points))
cands = gen_candidates(segs, len(segmentation(format_target_points(target))))

# get the total errs and the corresponding candidates
total_errs = dict()
for cand in cands:
    total_errs[total_err(shapeErr(format_target_points(target), desegmentation(cand)))] = cand

total_errs = sorted(total_errs.items(), key=lambda d:d[0])

err_delta = dict()

# Based on the top 10 candidates which have the most similar pattern to do anomaly detection
for i in range(10):
    delta = calc_delta(total_errs[i][1])
    mix = -0 * delta + 1 * total_errs[i][0]
    err_delta[mix] = total_errs[i][1]

err_delta = sorted(err_delta.items(), key=lambda d:d[0])


# plot the input data after exponential moving average and the top two candidates that are considered as anomaly
basedate = datetime(2016,1,1)
points = []
cnt = 2

for i, j in err_delta:
    if not cnt:
        break
    cnt -= 1
    rand_int = random.randint(1, 7)
    colors = ['r', 'b', 'g', "brown", "tomato", "tan", "gold", "olive"]
    for k in j:
        for point in k:
            dt = basedate
            for step in range(point.x):
                dt += timedelta(days=1)
            plt.scatter(dt, point.y, c=colors[rand_int], s=20)
            points.append([dt, point.y])

plt.plot(short_ema)
plt.title('Stock price for NFLX')
plt.xlabel('date')
plt.ylabel('price')
plt.show()
