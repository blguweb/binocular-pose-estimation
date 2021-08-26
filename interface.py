from data_loader import DSEC
import numpy as np
import cv2
import bisect
import os

class EstimateDepth:
    def __init__(self, path):
        self.dataset = DSEC(path)
        self.yita = 30 ### 30ms ###
        self.r, self.l = 480, 640

    def cal_time_surface(self, seq): ### 对某个事件系列进行计算！！！ ###
        t_last_all = {}
        start,end = np.min(seq[:, 0]), np.max(seq[:, 0])
        for (t, x, y, p) in seq:
            if (y, x) not in t_last_all.keys(): t_last_all[(y, x)] = [t]
            else: t_last_all[(y, x)].append(t)
        return t_last_all, start, end

    def get_image(self, t_last_all, cur_t):
        goal = np.zeros((self.r, self.l))
        for r in range(self.r):
            for l in range(self.l):
                if (r, l) in t_last_all.keys():
                    last_index = bisect.bisect_left(t_last_all[(r,l)], cur_t) - 1
                    cur_last = t_last_all[(r,l)][last_index]
                else: cur_last = 0
                goal[r, l] = np.exp(-(cur_t - cur_last) / self.yita)
        goal *= 255
        return goal.astype(np.uint8)

    def estimate(self):
        for index in range(len(self.dataset)):
            data = self.dataset.get_example(index)
            ts_left,s,t = self.cal_time_surface(data['left']['event_sequence'])
            ts_right,_,_ = self.cal_time_surface(data['right']['event_sequence'])
            left,right = self.get_image(ts_left, (s+t)/2), self.get_image(ts_right,(s+t)/2)
            cv2.imshow('left', left)
            cv2.imshow('right', right)
            cv2.waitKey(0)
            exit(0)


if __name__ == '__main__':
    estimateDepth = EstimateDepth('zurich_city_04_c')
    estimateDepth.estimate()