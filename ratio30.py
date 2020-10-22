import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt


def read_mat(paths, n_tasks, similar_index):
    acc = defaultdict(list)
    for seq_id, path in enumerate(paths):
        print (path)
#        print (len(similar_index[seq_id]))
        for t_id, t in enumerate(similar_index[seq_id]):
            forward = []
            with open(path) as sims:
                for sim_id,sim in enumerate(sims):
                    sim = sim.strip('\n').split('\t')
#                    print (sim_id, sim, t)
                    if sim_id == t:
                        forward.append(float(sim[t]))
#                    for i in range(t+1):
                    if sim_id < t and sim_id in similar_index[seq_id]:
                        forward.append(float(sim[sim_id]))
                    if t != sim_id: continue
                    
                    backward = []
                    for i in range(t+1):
                        if i in similar_index[seq_id]:
                            backward.append(float(sim[i]))
#                            acc['backward_'+str(t_id)].append(float(sim[i]))

                    acc['backward_'+str(t_id)].append(np.mean(backward))
                        
#                print (forward)
#                for x in forward:
                acc['forward_'+str(t_id)].append(np.mean(forward))
#                print (np.mean(forward), np.mean(backward))
                print('forward: ',forward)
                print('backward: ',backward)
    return acc




#def plot_data(acc, type):
#    values = []
#    steps = []
#
#    for key, value in acc.items():
#        if key.split('_')[0] == type:
##            print (key, len(acc[key]), np.mean(acc[key]))
#            values.append(np.mean(acc[key]))
#            steps.append(int(key.split('_')[1]))
#    plt.plot(steps, values)

def plot_data(accs, type):
    ratio = []
    acc, acc_one, acc_random = accs
    for key, value in acc_one.items():
        if key.split('_')[0] == type:
            norm = [acc[key][i] - acc_random[key][i] for i in range(len(acc_one[key]))]
            denorm = [acc_one[key][i] - acc_random[key][i] for i in range(len(acc_one[key]))]
            r = [norm[i] / (denorm[i]+1e-4) for i in range(len(norm))]
            ratio.append(np.mean(r))
    plt.plot(ratio)

def stat(ncl_path, one_path, random_path, mtcl_path, n_tasks, similar_index, title):
    data = []
    
    acc_ncl = read_mat(ncl_path, n_tasks, similar_index)
    # acc_one = read_mat(one_path, n_tasks, similar_index)
    # acc_random = read_mat(random_path, n_tasks, similar_index)
    # acc_mtcl = read_mat(mtcl_path, n_tasks, similar_index)
    #
    
    # plt.figure(1)
    #
    # plot_data([acc_ncl, acc_one, acc_random], 'forward')
    # plot_data([acc_mtcl, acc_one, acc_random], 'forward')
    #
    # plot_data([acc_ncl, acc_one, acc_random], 'backward')
    # plot_data([acc_mtcl, acc_one, acc_random], 'backward')
    #
    # plt.legend(['ncl forward', 'mtcl forward', 'ncl backward', 'mtcl backward'])
    # plt.savefig(title, bbox_inches='tight')
    # plt.clf()
    return


    plt.figure(1)
    plt.title(title)
    plot_data(acc_ncl, 'forward')
    plot_data(acc_ncl, 'backward')
    plot_data(acc_mtcl, 'forward')
    plot_data(acc_mtcl, 'backward')
    plot_data(acc_random, 'forward')
    plot_data(acc_random, 'backward')
    
    plot_data(acc_one, 'forward')

    plt.legend(['ncl forward', 'ncl backward', 'mtcl forward','mtcl backward', 'random forward','random backward', 'one forward'])
#    plt.show()
    plt.savefig(title, bbox_inches='tight')
    plt.clf()
#    for key, value in acc_ncl.items():
#        if key.split('_')[0] == 'forward':
#            print (key, np.mean(acc_ncl[key]))
#    for key, value in acc_ncl.items():e
#        if key.split('_')[0] == 'backward':
#            print (key, np.mean(acc_ncl[key]))
#
#    for key, value in acc_one.items():
#        if key.split('_')[0] == 'forward':
#            print (key, np.mean(acc_one[key]))
#


if __name__ == '__main__':

    # n_tasks = [30, 30]
    n_tasks = [30,]

    ncl = [
        [
        'res/mannual/mixemnist30-small-small/mixemnist_sgd_0_random0,small,ntask30.txt',
        'res/mannual/mixemnist30-small-small/mixemnist_sgd_0_random1,small,ntask30.txt',
        'res/mannual/mixemnist30-small-small/mixemnist_sgd_0_random2,small,ntask30.txt',
        'res/mannual/mixemnist30-small-small/mixemnist_sgd_0_random3,small,ntask30.txt',
        'res/mannual/mixemnist30-small-small/mixemnist_sgd_0_random4,small,ntask30.txt'
        ],
        # [
        # 'res30/mlpmixceleba_sgd_0_random0,small,ntask30.txt',
        # 'res30/mlpmixceleba_sgd_0_random1,small,ntask30.txt',
        # 'res30/mlpmixceleba_sgd_0_random2,small,ntask30.txt',
        # 'res30/mlpmixceleba_sgd_0_random3,small,ntask30.txt',
        # 'res30/mlpmixceleba_sgd_0_random4,small,ntask30.txt',
        # ]
    ]
    one = [
        [
        'res/mannual/mixemnist30-small-small/mixemnist_sgd-restart_0_random0,small,ntask30.txt',
        'res/mannual/mixemnist30-small-small/mixemnist_sgd-restart_0_random1,small,ntask30.txt',
        'res/mannual/mixemnist30-small-small/mixemnist_sgd-restart_0_random2,small,ntask30.txt',
        'res/mannual/mixemnist30-small-small/mixemnist_sgd-restart_0_random3,small,ntask30.txt',
        'res/mannual/mixemnist30-small-small/mixemnist_sgd-restart_0_random4,small,ntask30.txt'
        ],
        # [
        # 'res30/mlpmixceleba_sgd-restart_0_random0,small,ntask30.txt',
        # 'res30/mlpmixceleba_sgd-restart_0_random1,small,ntask30.txt',
        # 'res30/mlpmixceleba_sgd-restart_0_random2,small,ntask30.txt',
        # 'res30/mlpmixceleba_sgd-restart_0_random3,small,ntask30.txt'
        # 'res30/mlpmixceleba_sgd-restart_0_random4,small,ntask30.txt'
        # ]
    ]

    hat = [
        [
        'res/mannual/mixemnist30-small-small/mixemnist_hat_0_random0,small,ntask30.txt',
        'res/mannual/mixemnist30-small-small/mixemnist_hat_0_random1,small,ntask30.txt',
        'res/mannual/mixemnist30-small-small/mixemnist_hat_0_random2,small,ntask30.txt',
        'res/mannual/mixemnist30-small-small/mixemnist_hat_0_random3,small,ntask30.txt',
        'res/mannual/mixemnist30-small-small/mixemnist_hat_0_random4,small,ntask30.txt'
        ],
        # [
        # 'res30/mlpmixceleba_sgd-restart_0_random0,small,ntask30.txt',
        # 'res30/mlpmixceleba_sgd-restart_0_random1,small,ntask30.txt',
        # 'res30/mlpmixceleba_sgd-restart_0_random2,small,ntask30.txt',
        # 'res30/mlpmixceleba_sgd-restart_0_random3,small,ntask30.txt'
        # 'res30/mlpmixceleba_sgd-restart_0_random4,small,ntask30.txt'
        # ]
    ]

    mtcl = [
        [
        'res/self-small-small/ntasks30/mixemnist_MixKan_0_random0,small,auto,multi-loss-joint-Tsim,ntask30,pdrop0.2,lr0.025,patient10,nhead5.txt_mcl',
        'res/self-small-small/ntasks30/mixemnist_MixKan_0_random1,small,auto,multi-loss-joint-Tsim,ntask30,pdrop0.2,lr0.025,patient10,nhead5.txt_mcl',
        'res/self-small-small/ntasks30/mixemnist_MixKan_0_random2,small,auto,multi-loss-joint-Tsim,ntask30,pdrop0.2,lr0.025,patient10,nhead5.txt_mcl',
        'res/self-small-small/ntasks30/mixemnist_MixKan_0_random3,small,auto,multi-loss-joint-Tsim,ntask30,pdrop0.2,lr0.025,patient10,nhead5.txt_mcl',
        'res/self-small-small/ntasks30/mixemnist_MixKan_0_random4,small,auto,multi-loss-joint-Tsim,ntask30,pdrop0.2,lr0.025,patient10,nhead5.txt_mcl',
        ],
        # [
        # 'mannual/mtcl/mlpmixceleba_MixKan_0_random0,small,auto,multi-loss-joint-Tsim,lr0.025,patient10,nhead5.txt_mcl',
        # 'mannual/mtcl/mlpmixceleba_MixKan_0_random1,small,auto,multi-loss-joint-Tsim,lr0.025,patient10,nhead5.txt_mcl',
        # 'mannual/mtcl/mlpmixceleba_MixKan_0_random2,small,auto,multi-loss-joint-Tsim,lr0.025,patient10,nhead5.txt_mcl',
        # 'mannual/mtcl/mlpmixceleba_MixKan_0_random3,small,auto,multi-loss-joint-Tsim,lr0.025,patient10,nhead5.txt_mcl',
        # 'mannual/mtcl/mlpmixceleba_MixKan_0_random4,small,auto,multi-loss-joint-Tsim,lr0.025,patient10,nhead5.txt_mcl'
        # ]
    ]

    
    # random = [
    #     [
    #      'res30/mixemnist_sgd_0_random0,small,ntask30.txt',
    #      'res30/mixemnist_sgd_0_random1,small,ntask30.txt',
    #      'res30/mixemnist_sgd_0_random2,small,ntask30.txt',
    #      'res30/mixemnist_sgd_0_random3,small,ntask30.txt',
    #      'res30/mixemnist_sgd_0_random4,small,ntask30.txt'
    #     ],
    #     # [
        #  'res30/mlpmixceleba_random_0_random0,mixceleba,small,ntasks30.txt',
        #  'res30/mlpmixceleba_random_0_random1,mixceleba,small,ntasks30.txt',
        #  'res30/mlpmixceleba_random_0_random2,mixceleba,small,ntasks30.txt',
        #  'res30/mlpmixceleba_random_0_random3,mixceleba,small,ntasks30.txt',
        #  'res30/mlpmixceleba_random_0_random4,mixceleba,small,ntasks30.txt'
        #  ]
    # ]
    
    similar_index = [
        [
        [2,7,10,12,14,17,20,22,27,29],
        [1,7,8,9,10,12,21,25,27,28],
        [2,3,4,6,10,13,15,23,28,29],
        [5,6,9,14,20,21,24,25,26,27],
        [1,6,12,16,20,22,23,27,28,29]
        ],
        # [
        # [2,4,6,10,17,20,21,27,28,29],
        # [1,2,5,13,14,16,18,19,20,26],
        # [3,10,11,12,13,19,20,24,28,29],
        # [0,3,7,8,9,13,14,19,21,29],
        # [0,1,2,3,4,5,7,15,22,24]
        # ]
    ]
    
    
    dissimilar_index = []
    for i in range(len(similar_index)):
        s = similar_index[i]
        nt = n_tasks[i]
        a = []
        for j in range(len(s)):
            b = []
            for k in range(nt):
                if k not in s[j]:
                    b.append(k)
            a.append(b)
        dissimilar_index.append(a)

    # for i in range(len(similar_index)):
    #    for j in range(len(similar_index[i])):
    #        print (similar_index[i][j], dissimilar_index[i][j])

    id = 0; stat(ncl[id], one[id], hat[id], mtcl[id], n_tasks[id], similar_index[id], 'fe-mnist-30')
#    id = 1; stat(ncl[id], one[id], random[id], ncl[id], n_tasks[id], similar_index[id], 'fe-celeba-30')

    # id = 0; stat(ncl[id], one[id], random[id], mtcl[id], n_tasks[id], dissimilar_index[id], 'emnist-30')
#    id = 1; stat(ncl[id], one[id], random[id], ncl[id], n_tasks[id], dissimilar_index[id], 'cifar-30')


















#
#tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
#             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
#             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
#             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
#             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
#
#def sliding_mean(data_array, window=5):
#    data_array = np.array(data_array)
#    new_list = []
#    for i in range(len(data_array)):
#        indices = range(max(i - window + 1, 0),min(i + window + 1, len(data_array)))
#        avg = 0
#        for j in indices:
#            avg += data_array[j]
#        avg /= float(len(indices))
#        new_list.append(avg)
#    return np.array(new_list)
#
#def csvplot(title, values, steps, xr, yr, ytick, ypos):
#    print (values)
#    print (steps)
#    plt.figure(figsize=(11, 3))
#    ax = plt.subplot(111)
#    ax.spines["top"].set_visible(False)
#    ax.spines["bottom"].set_visible(False)
#    ax.spines["right"].set_visible(False)
#    ax.spines["left"].set_visible(False)
#    ax.get_xaxis().tick_bottom()
#    ax.get_yaxis().tick_left()
#
#    # Limit the range of the plot to only where the data is.
#    # Avoid unnecessary whitespace.
#    plt.xlim(xr[0], xr[1])
#    plt.ylim(yr[0], yr[1])
#    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
#    #    for y in ytick[:-1]:
#    #        plt.plot(range(0,30001,5000), [y] * len(range(0,30001,5000)), 'k:', lw=0.5, alpha=0.3)
#
#    # Remove the tick marks; they are unnecessary with the tick lines we just plotted.
#    plt.tick_params(axis="both", which="both", bottom="off", top="off",labelbottom="on", left="on", right="off", labelleft="on")
#    cid = 0*2 % 8
#
#    window_size = 2 # 0.01 10 over 1000
#    mean = sliding_mean(values, window=window_size)
#    err = np.abs(values - mean)*1.6
#    err = sliding_mean(err, window=1)
#    err_c = list(tableau20[cid+1]) + [0.3]
#    plt.fill_between(steps, mean-err, mean+err, color=err_c)
#    plt.plot(steps, mean, lw=2.5, color=tableau20[cid])
#    y_pos = mean[-1] + ypos[i]
#    #        plt.text(100500, y_pos, label[i], fontsize=10, color=tableau20[cid], ha="left")
#    plt.savefig(title+'.png', bbox_inches="tight") # pdf: font-embedding problem
#    plt.show()




#    xr = [0,30]
#    title = 'T_l2_loss'
#    yr = [0.0, 1]
#    ytick = [0.0, 1]
#    ypos = [
#         0,
#         0,
#         0,
#         0,
#    ]
##    label = [
##          # r'$\mathcal{L}_d: f$',
##          # r'$\mathcal{L}_d: fl$',
##          r'$\mathcal{L}_s: f$',
##          r'$\mathcal{L}_s: fl$',
##          ]
#    csvplot(title, values, steps, xr, yr, ytick, ypos)



