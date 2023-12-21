from pathlib import Path
import os.path
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
import matplotlib.lines as mlines

plt.rcParams['font.sans-serif'] = ['Times New Roman']  # font


data_path = Path(r'/data')
dirs_different = [r'.\data\different\T2=0.6',
                  r'.\data\different\T2=1',
                  r'.\data\different\T2=5.5']
Psucc_file = 'Psucc.mat'

fmt_list_diff = ['^', 's', 'd']  # marker shape
labels_diff = [r'$T_2=0.6s$', r'$T_2=1.0s$', r'$T_2=5.4s$']  # legend label
colors_diff = ['#24943b', '#1a2d72', '#921e23']  # marker color

fig, (ax_diff) = plt.subplots(1, 1, figsize=(7.57, 5.35))
# for f in data_path.rglob('*.mat'):
for idx in range(len(dirs_different)):
    Psucc_mat = sp.io.loadmat(os.path.join(dirs_different[idx], Psucc_file))
    if 'Psucc_mean' in Psucc_mat:
        Psucc_mean = Psucc_mat['Psucc_mean'].flatten()
        Psucc_std = Psucc_mat['Psucc_std'].flatten()
    else:
        Psucc_mean = Psucc_mat['Psucc_ave'].flatten()
        Psucc_std = Psucc_mat['Psucc_std'].flatten()
    t = np.arange(13) / 6  # np.arange generate [0, 13)
    ax_diff.errorbar(t, Psucc_mean, Psucc_std, fmt=fmt_list_diff[idx], label=labels_diff[idx], capsize=2,
                     color=colors_diff[idx])

plt.xlim([0, 2.03])
plt.ylim([0.497, 0.66])
plt.xticks([0, 0.5, 1.0, 1.5, 2.0], fontsize=18, fontweight='medium')
plt.yticks([0.50, 0.54, 0.58, 0.62, 0.66], fontsize=18, fontweight='medium')
plt.tick_params(direction='in', length=5, width=1)
plt.xlabel('Encoding time (s)', fontsize=20, fontweight='medium')
plt.ylabel('Successful probability', fontsize=20, fontweight='medium')
handles, labels = ax_diff.get_legend_handles_labels()
# remove the errorbars
handles = [h[0] for h in handles]
green_line, blue_line, red_line = (mlines.Line2D([], [], color=color) for color in colors_diff)
handles_tuple = tuple(handles)
handles[0] = (green_line, handles[0])  # add a line before marker
handles[1] = (blue_line, handles[1])
handles[2] = (red_line, handles[2])
handles.append(handles_tuple)  # legend for 'Exp.data'
labels.append('Exp.data')
ax_diff.legend(handles, labels, loc='upper left', markerscale=0.8, framealpha=0,
               fontsize=12,
               handler_map={tuple: HandlerTuple(ndivide=None)})


# fill area between curves
def poly_curve(x, a=1.0, b=0.0, c=0.0, d=0.0):
    return a*x**3 + b*x**2 + c*x + d


def color_norm(x):
    x1 = x - np.mean(x)
    x1 = np.abs(x1) / np.max(x1)
    return 1/(1 + np.exp(6 * (x1 - 0.95)))


x = np.linspace(0, 2, 50)
y1 = poly_curve(x, a=0.0054, b=-0.0739, c=0.1919, d=0.4976)
y2 = poly_curve(x, a=-0.0101, b=-0.018, c=0.1336, d=0.4929)
y_ref = 0.09*x + 0.5
plt.fill_between(x, y1, y_ref, where=y1 > y_ref, color='#bae0dc', alpha=color_norm(x[y1>y_ref]), linewidth=0)
plt.fill_between(x, y2, y_ref, where=y2 > y_ref, color='#aac1e1', alpha=color_norm(x[y2>y_ref]), linewidth=0)
plt.plot(x, y1, color='#24943b')
plt.plot(x, y2, color='#1a2d72')
plt.plot(x, y_ref, color='black', linestyle='dashed')
plt.text(1.2, 1.2*0.09+0.505, 'Unitary dynamics', fontsize=17,
         rotation=np.arctan(0.09)*180/np.pi, transform_rotates_text=True, rotation_mode='anchor')

plt.tight_layout()


# add nested plot
def s_curve(x, a=1/0.18, b=4.0, c=1.2):
    return 1/(a + np.exp(-1*b*(x - c)))


x = np.linspace(0, 4, 100)
fig.add_axes((0.6934, 0.2614, 0.2188, 0.2432))
plt.plot(x, s_curve(x), label=r'$T_1=0.01$ s', color='#24943b')
plt.plot(x, s_curve(x, b=5, c=1.3), label=r'$T_1=0.1$ s', color='#1a2d72')
plt.plot(x, s_curve(x, b=7, c=1.4), label=r'$T_1=1$ s', color='#921e23')
plt.plot(x, s_curve(x, b=10, c=1.5), label=r'$T_1=10$ s', color='black')
plt.xlim([0, 4.1])
plt.ylim([0, 0.185])
plt.xticks([0, 1, 2, 3, 4], fontsize=12)
plt.yticks([0, 0.06, 0.12, 0.18], fontsize=12)
plt.tick_params(direction='in', length=1.5, width=1, top=True, right=True)
plt.xlabel(r'$log_{10}(T_1/T_2)$', fontsize=12)
plt.ylabel(r'$\eta$', fontsize=12)
plt.legend(loc='lower right', framealpha=0, fontsize='small', handlelength=1)

plt.savefig('Psucc.eps')
plt.show()
