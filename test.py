from data_loader import DSEC
from metric import compute_relative_depth_error
import numpy as np
import cv2
import bisect
import os
import time

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
font_pro = FontProperties(fname='C:/Windows/Fonts/STKAITI.TTF', size=12)
font_pro_min = FontProperties(fname='C:/Windows/Fonts/STKAITI.TTF', size=10)
font_pro_max = FontProperties(fname='C:/Windows/Fonts/STKAITI.TTF', size=18)
from mpl_toolkits.mplot3d import axes3d
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
plt.rcParams['xtick.direction'] = 'in'  # in; out; inout
plt.rcParams['ytick.direction'] = 'in'
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import axes3d
from tqdm import tqdm
from sample import plane_sweep_ncc
from sgbm import sgbm
from bm import bm


class EstimateDepth:
    def __init__(self, path, which):
        self.dataset = DSEC(path)
        self.yita = 30 ### 30ms ###
        self.r, self.l = 480, 640
        self.count = 1 ### 对于每张图片 将所有时间节点分成5等份 对每个等分来求解 ###
        self.M = 4
        self.every = 10
        self.which = which
        self.path = path

    def show(self, d):
        img = np.ones((480, 640, 3))
        img *= 255
        t_cur, _, _, _ = d[0]
        for ii in range(d.shape[0]):
            t, x, y, p = d[ii]
            x, y = int(x), int(y)
            if p == 1:
                img[y, x] = [0, 0, 255]
            else:
                img[y, x] = [255, 0, 0]
        return img.astype(np.uint8)

    def test_ts(self, t_last_all, cur_t, ax):
        goal = np.zeros((self.r, self.l))
        for r in range(self.r):
            for l in range(self.l):
                if (r, l) in t_last_all.keys():
                    last_index = bisect.bisect_left(t_last_all[(r,l)], cur_t) - 1
                    cur_last = t_last_all[(r,l)][last_index]
                else: cur_last = 0
                goal[r, l] = np.exp(-(cur_t - cur_last) / self.yita)
        goal *= 255
        return ax.imshow(goal, cmap=plt.cm.jet)

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

    def cal_time_surface(self, seq): ### 对某个事件系列进行计算！！！ ###
        t_last_all = {}
        start,end = np.min(seq[:, 0]), np.max(seq[:, 0])
        for (t, x, y, p) in seq:
            if (y, x) not in t_last_all.keys(): t_last_all[(y, x)] = [t]
            else: t_last_all[(y, x)].append(t)
        return t_last_all, start, end

    def visulize_data(self):
        pbar = tqdm(range(50))
        fig,ax = plt.subplots()
        ims = []
        for index in pbar:
            data = self.dataset.get_example(index)
            ts_left, s, t = self.cal_time_surface(data['left']['event_sequence'])
            ts_right, s1, t1 = self.cal_time_surface(data['right']['event_sequence'])
            goal = self.test_ts(ts_left, (s+t)/2, ax)
            ims.append([goal])
        ani = animation.ArtistAnimation(fig, ims, interval=100)
        ani.save("test.gif")

    def img2txt(self, img, to_path): ### for patchmatch ###
        data = [img.shape[0], img.shape[1]]
        for cc in range(3):
            for ii in range(img.shape[0]):
                for kk in range(img.shape[1]):
                    if cc == 0: data.append(img[ii, kk])
                    else: data.append(0) ### because only one channel ###
        with open(to_path, 'w') as fp:
            data = [str(d) + ' ' for d in data]
            fp.writelines(data)
            fp.close()

    def show_res(self, path, left, right): ### for patchmatch ###
        fig, ax = plt.subplots()
        img = np.loadtxt(path)
        rows, cols = int(img[0]), int(img[1])
        img = img[2:].reshape((rows, cols, 2)).astype(np.int)
        x,y = 100, 20
        width = 20
        ims = []
        total = 20
        for ii in range(total):
            print(ii)
            y += self.l // (total+1)
            l,r = cv2.cvtColor(left, cv2.COLOR_GRAY2BGR), cv2.cvtColor(right, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(l, (y, x), (y+width,x+width), (255,0,0), 5)
            cv2.rectangle(r, (img[x,y,1], img[x,y,0]), (img[x,y,1]+width, img[x,y,0]+width), (255,0,0), 5)
            target = np.zeros((self.r, 2*self.l, 3), dtype = np.uint8)
            target[:, :self.l, :] = l
            target[:, self.l:, :] = r
            im = ax.imshow(target)
            ims.append([im])
        ani = animation.ArtistAnimation(fig, ims, interval=500)
        ani.save("res/patchmatch.gif")

    def get_disp(self):
        img = np.loadtxt('resources/data/para_answers/ans1.txt')
        rows, cols = int(img[0]), int(img[1])
        img = img[2:].reshape((rows, cols, 2)).astype(np.int)
        return img[:, :, 1]



    def caculate_error(self, sequence, off, ground_truth, baseline):
        final_off = np.zeros((sequence.shape[0], 5))
        for ii in range(sequence.shape[0]):
            t,x,y,p = sequence[ii]
            row,col = int(y),int(x)
            if self.which == 'patchmatch':
                final_off[ii, :] = [t, int(x), int(y), p, abs(off[row, col] - col)] ### keng ###
            else: final_off[ii, :] = [t, int(x), int(y), p, abs(off[row, col])] ### keng ###
        # final_off[:,-1] = 10 + 20 * (final_off[:, -1] / np.max(final_off[:, -1]))
        clip = final_off[:, -1] <= 3
        final_off[clip, -1] = 10
        clip = final_off[:, -1] >= 30
        final_off[clip, -1] = 30
        return compute_relative_depth_error(final_off, ground_truth, baseline, depth_range=[5, 50])

    def all_metric(self, path):
        file = os.listdir(path)
        all_error = []
        index = 0
        for f in file:
            p = path + '/' + f
            index = int(f[:f.index('.')])
            data = self.dataset.get_example(index)
            off = np.loadtxt(p).reshape((self.r, self.l))
            error = self.caculate_error(data['left']['event_sequence'], off, data['left']['disparity_image'],
                                        data['baseline'])
            print(p, error, index)
            all_error.append(error)
            index += 1
            if index % 10 == 0: np.savetxt(path+'.txt', np.array(all_error), fmt='%.5f', delimiter=' ')

    def find_best_patch(self, left, right):
        disp = np.zeros((self.r, self.l))
        error = np.zeros((self.r - self.M, self.l - self.M))
        def cal_D(i, k, ii, kk):
            #print(left[i:i+self.M, k:k+self.M] * right[ii:ii+self.M, kk:kk+self.M])
            return np.count_nonzero(left[i:i+self.M, k:k+self.M] * right[ii:ii+self.M, kk:kk+self.M])
            #return cv2.norm(left[i:i+self.M, k:k+self.M] - right[ii:ii+self.M, kk:kk+self.M], cv2.NORM_L1)
        for ii in range(self.r-self.M):
            error[ii, 0] = cal_D(ii, 0, ii, 0)
            for kk in range(1, self.l-self.M): ### 只需要寻找水平匹配 ###
                # print(int(kk-1-disp[ii, kk-1]), kk)
                start = int(kk-1-disp[ii, kk-1])
                for jj in range(start, min(kk, start + 3*self.M)): ### 从上一个点开始 ###
                    d = cal_D(ii, kk, ii, jj)
                    if d >= error[ii, kk]:
                        error[ii, kk] = d
                        disp[ii, kk] = kk-jj
        return disp

    def method_2(self,image):
        blurred = cv2.GaussianBlur(image, (3, 3), 0)
        t, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        return binary

    def show_ground_truth(self, left, right, gt):
        fig, ax = plt.subplots()
        x, y = 100, 20
        width = 20
        ims = []
        total = 20
        for ii in range(total):
            print(ii)
            y += self.l // (total + 1)
            l, r = cv2.cvtColor(left, cv2.COLOR_GRAY2BGR), cv2.cvtColor(right, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(l, (y, x), (y + width, x + width), (255, 0, 0), 5)
            if gt[x, y] != np.inf: gt_y = max(y - gt[x, y], 0)
            else: gt_y = y
            cv2.rectangle(r, (int(gt_y), int(x)), (int(gt_y+width), int(x+width)), (255, 0, 0), 5)
            target = np.zeros((self.r, 2 * self.l, 3), dtype=np.uint8)
            target[:, :self.l, :] = l
            target[:, self.l:, :] = r
            im = ax.imshow(target)
            ims.append([im])
        ani = animation.ArtistAnimation(fig, ims, interval=1000)
        ani.save("res/gt.gif")

    def estimate(self, test = 'random'):
        which = self.which
        random_index = np.arange(len(self.dataset))
        if test == 'random': np.random.shuffle(random_index)
        else: self.every = len(self.dataset)
        all_time = []
        for ii in range(self.every):
            print('第', ii, '个', random_index[ii], len(self.dataset))
            index = random_index[ii]
            data = self.dataset.get_example(index)
            ts_left,s,t = self.cal_time_surface(data['left']['event_sequence'])
            ts_right,_,_ = self.cal_time_surface(data['right']['event_sequence'])
            pre_disp,total = np.zeros((self.r, self.l)) , 0
            for ii in range(self.count):
                start = time.time()
                left,right = self.get_image(ts_left, (s+t)/2), self.get_image(ts_right, (s+t)/2)
                left, right = self.method_2(left), self.method_2(right)
                self.show_ground_truth(left, right, data['left']['disparity_image'])
                exit(0)
                if which == 'patchmatch':
                    self.img2txt(left, 'resources/data/img2txts/A1.txt')
                    self.img2txt(right, 'resources/data/img2txts/B1.txt')
                    os.system(os.getcwd() + r'/resources/para_patchmatch.exe')
                    pre_disp += self.get_disp()
                elif which == 'sgbm': pre_disp += sgbm(left, right)
                elif which == 'bm': pre_disp += bm(left, right)
                total += 1
                print(time.time() - start)
                all_time.append(time.time() - start)
            pre_disp /= total
            np.savetxt('answer/'+str(index)+'.txt', pre_disp, delimiter=' ', fmt='%.5f')
        np.savetxt('time_consume.txt', np.array(all_time), fmt='%.5f')

    def first_show(self):
        data = self.dataset.get_example(0)
        l = self.show(data['left']['event_sequence'])
        r = self.show(data['right']['event_sequence'])
        l = cv2.copyMakeBorder(l, 10,10,10,10, borderType=cv2.BORDER_DEFAULT)
        cv2.imwrite('thuna_left.png', l)
        cv2.imwrite('thuna_right.png', r)

# self.show_res('resources/data/para_answers', left, right)
# exit(0)
if __name__ == '__main__':
    estimateDepth = EstimateDepth('zurich_city_04_c', 'patchmatch')
    estimateDepth.estimate(test = 'all')
    # estimateDepth.all_metric('final_answers/answer_sgbm_interc')

    #estimateDepth.first_show()
    # a = [1,1,1,1,1, 2,2,2,3]
    # print(bisect.bisect_left(a, 2.5))
    # exit(0)
    # self.show_res('resources/data/para_answers/ans1.txt', left, right)
    # dst = self.method_2(left)
    # cv2.imshow('left', left)
    # cv2.imshow('dst', dst)
    # cv2.waitKey(0)