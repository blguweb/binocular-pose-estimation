from data_loader import DSEC
from metric import compute_relative_depth_error
import numpy as np
import cv2
import bisect
import os

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

def draw_pic(path, label):
    error = np.loadtxt(path)
    avg = np.average(error)
    plt.plot(range(len(error)), error, c = 'red', label = 'Value of Error')
    plt.plot([0, len(error)], [avg, avg], c = 'blue', linewidth = 5, label = 'Average {:.2f}'.format(avg))

    plt.xlabel('共计'+str(len(error))+'组数据', fontproperties = font_pro)
    plt.ylabel('每个测试点的error值', fontproperties=font_pro)
    plt.legend()
    if save: plt.savefig('res/'+label+'.png', dpi=500, bbox_inches='tight')
    plt.show()
    plt.close()

    plt.boxplot(error, vert=False, showfliers = False, patch_artist='green', widths=0.7)
    plt.xlabel('Relative Error', fontproperties = font_pro)
    if save: plt.savefig('res/box_' + label + '.png', dpi=500, bbox_inches='tight')
    plt.show()
    plt.close()

def draw_time(path, label):
    error = np.loadtxt(path)
    avg = np.average(error)
    plt.plot(range(len(error)), error, c='red', label='Value of Error')
    plt.plot([0, len(error)], [avg, avg], c='blue', linewidth=5, label='Average {:.2f}'.format(avg))

    plt.xlabel('共计' + str(len(error)) + '组数据', fontproperties=font_pro)
    plt.ylabel('每个测试点的运行时长(s)', fontproperties=font_pro)
    plt.legend()
    if save: plt.savefig('res/' + label + '.png', dpi=500, bbox_inches='tight')
    plt.show()
    plt.close()

    plt.boxplot(error, vert=False, showfliers=False, patch_artist='green', widths=0.7)
    plt.xlabel('运行时长(s)', fontproperties=font_pro)
    if save: plt.savefig('res/box_' + label + '.png', dpi=500, bbox_inches='tight')
    plt.show()
    plt.close()

def draw_all():
    p_data = np.loadtxt('final_answers/answer_patchmatch_thuna.txt')
    p1_data = np.loadtxt('final_answers/answer_patchmatch_interc.txt')
    patchmatch_data = np.append(p_data, p1_data, axis=0)

    p_data = np.loadtxt('final_answers/answer_bm_thun1.txt')
    p1_data = np.loadtxt('final_answers/answer_bm_interc1.txt')
    bm_data = np.append(p_data, p1_data, axis=0)

    p_data = np.loadtxt('final_answers/answer_sgbm_thun.txt')
    p1_data = np.loadtxt('final_answers/answer_sgbm_interc.txt')
    sgbm_data = np.append(p_data, p1_data, axis=0)

    plt.plot(range(len(patchmatch_data)), patchmatch_data, c='red', label='PatchMatch Error')
    plt.plot(range(len(bm_data)), bm_data, c='blue', label='Bm Error')
    plt.plot(range(len(sgbm_data)), sgbm_data, c='brown', label='Sgbm Error')

    plt.xlabel('共计' + str(len(patchmatch_data)) + '组数据', fontproperties=font_pro)
    plt.ylabel('Relative Error', fontproperties=font_pro)
    plt.legend()
    if save: plt.savefig('res/all.png', dpi=500, bbox_inches='tight')
    plt.show()
    plt.close()

    plt.boxplot([patchmatch_data, bm_data, sgbm_data], showfliers=False, patch_artist='green',
                labels=['PatchMatch', 'BM', 'SGBM'])
    plt.ylabel('Relative Error', fontproperties=font_pro)
    if save: plt.savefig('res/box_all.png', dpi=500, bbox_inches='tight')
    plt.show()
    plt.close()

def draw_all_time():
    patchmatch_data = np.loadtxt('patchmatch_time.txt')
    bm_data = np.loadtxt('bm_time.txt')
    sgbm_data = np.loadtxt('sgbm_time.txt')

    plt.plot(range(len(patchmatch_data)), patchmatch_data, c='red', label='PatchMatch')
    plt.plot(range(len(bm_data)), bm_data, c='blue', label='Bm')
    plt.plot(range(len(sgbm_data)), sgbm_data, c='brown', label='Sgbm')

    plt.xlabel('共计' + str(len(patchmatch_data)) + '组数据', fontproperties=font_pro)
    plt.ylabel('运行时长(s)', fontproperties=font_pro)
    plt.legend()
    if save: plt.savefig('res/all_time.png', dpi=500, bbox_inches='tight')
    plt.show()
    plt.close()

    plt.boxplot([patchmatch_data, bm_data, sgbm_data], showfliers=False, patch_artist='green',
                labels=['PatchMatch', 'BM', 'SGBM'])
    plt.ylabel('运行时长(s)', fontproperties=font_pro)
    if save: plt.savefig('res/box_all_time.png', dpi=500, bbox_inches='tight')
    plt.show()
    plt.close()


if __name__ == '__main__':
    save = True
    # draw_pic('final_answers/answer_patchmatch_thuna.txt', label = 'patchmatch_thuna_plot')

    # draw_pic('final_answers/answer_sgbm_thun.txt', label='sgbm_thun')

    # draw_pic('final_answers/answer_bm_thun1.txt', label='bm_thun1')

    # draw_pic('final_answers/answer_patchmatch_interc.txt', label='patchmatch_interc')

    # draw_pic('final_answers/answer_sgbm_interc.txt', label='sgbm_interc')

    # draw_pic('final_answers/answer_bm_interc1.txt', label='bm_interc1')

    # draw_time('patchmatch_time.txt', 'para_time')
    # draw_time('sgbm_time.txt', 'sgbm_time')
    # draw_time('bm_time.txt', 'bm_time')

    draw_all()
    # draw_all_time()

