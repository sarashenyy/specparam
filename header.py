from astropy.io import fits
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.transforms as mtransforms
import random
from matplotlib_venn import venn3, venn2
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter


# 去除没有 gaia_source_id 的数据
def getData_ingaia(file):
    cond_nan_gaiaid = np.isnan(file['gaia_source_id']) | (file['gaia_source_id'] < -9000)
    ingaia_index = set(file.index) - set(np.where(cond_nan_gaiaid)[0])
    print('total:', len(file.index))
    print('WITHOUT gaia source:', len(np.where(cond_nan_gaiaid)[0]))
    print('WITH gaia source:', len(ingaia_index))

    return_file = file.loc[ingaia_index, :].reset_index(drop=True)
    return return_file


# 按照 APOGEE 筛选类太阳星
def solar_like_APOGEE(file, teff_min=4800, teff_max=6800, feh_min=-1.0, feh_max=1.0, teff_name='TEFF_1',
                      logg_name='LOGG_1', feh_name='FE_H'):
    cond_teff = (file[teff_name] >= teff_min) & (file[teff_name] <= teff_max)
    cond_logg = (file[logg_name] >= (5.98 - 0.00035 * file[teff_name]))
    cond_feh = (file[feh_name] >= feh_min) & (file[feh_name] <= feh_max)
    # cond_flag = (file['STARFLAG']==0) & (file['ASPCAPFLAG']==0)

    # return_file = file.loc[ np.where( cond_teff & cond_logg & cond_feh & cond_flag)[0],: ].reset_index(drop=True)
    return_file = file.loc[np.where(cond_teff & cond_logg & cond_feh)[0], :].reset_index(drop=True)
    print('number of LAMOST solar-like dataset stars:', file.shape[0])
    # print('number of LAMOST-APOGEE solar-like stars:',len(np.where( cond_teff & cond_logg & cond_feh & cond_flag)[0]) )
    print('number of LAMOST-APOGEE solar-like stars:', len(np.where(cond_teff & cond_logg & cond_feh)[0]))

    return return_file


def BpRp_cut(data):
    cond_bprp = ((data['bp_rp'] - data['ebpminrp_gspphot']) > 0) & ((data['bp_rp'] - data['ebpminrp_gspphot']) < 1.5)
    index = np.where(cond_bprp)[0]
    return_data = data.loc[index, :].reset_index(drop=True)
    print('bprp and ebpminrp_gspphot not nan and bprp0 in [0,1.5]: ', len(np.where(cond_bprp)[0]))
    return return_data


# 统计拥有不可信 parallax 和 distance 数据的源并去除,correction=True表示视差取经过零点校正的数据
# 画韦恩图对上述结果可视化
def resonable_dis_plx(file, l_p_limit=5, correction=False):
    # 用 ｜ 进行或运算（取并集）时注意括号很重要！！ A | (b>0) 中的括号丢了就会计算错误！！
    cond_nan_d_gsp = (np.isnan(file['distance_gspphot'])) | (file['distance_gspphot'] <= 0)
    cond_nan_d_jones = (np.isnan(file['r_med_geo'])) | (file['r_med_geo'] <= 0) | (np.isnan(file['r_med_photogeo'])) | (
                file['r_med_photogeo'] <= 0)

    if correction == True:
        cond_nan_p = (np.isnan(file['plx_correction'])) | (file['plx_correction'] < 0)
        cond_low_p = (np.isnan(file['plx_correction_oe'])) | (file['plx_correction_oe'] <= l_p_limit)
    elif correction == False:
        cond_nan_p = (np.isnan(file['parallax'])) | (file['parallax'] < 0)
        cond_low_p = (np.isnan(file['parallax_over_error'])) | (file['parallax_over_error'] <= l_p_limit)

    no_dgsp = set(np.where(cond_nan_d_gsp)[0])
    no_jones = set(np.where(cond_nan_d_jones)[0])
    no_p = set(np.where(cond_nan_p)[0])
    low_p = set(np.where(cond_low_p)[0])

    print('no distance from GSP-Phot in sample:', len(no_dgsp))
    print('no distance from Bailer-Jones in sample:', len(no_jones))
    if correction == True:
        print('no parallax(AFTER zeropoint correction) in sample (including nan and negative):', len(no_p))
    elif correction == False:
        print('no parallax(WITHOUT zeropoint correction) in sample (including nan and negative):', len(no_p))
    print('no distance from GSP-Phot and Bailer-Jones and parallax:', len(no_dgsp | no_jones | no_p), '\n')

    if correction == True:
        print('low parallax quality in sample (plx_correction_oe <= ', l_p_limit, '):', len(low_p - no_p))
    elif correction == False:
        print('low parallax quality in sample (parallax_over_error <= ', l_p_limit, '):', len(low_p - no_p))
    print('bad distance or parallax in total:', len(no_dgsp | no_jones | no_p | low_p))

    plt.figure(figsize=(6, 6))
    venn3(
        subsets=[no_dgsp, no_p, low_p],
        set_labels=('A', 'B', 'C'),
        set_colors=('r', 'b', 'g')
    )
    plt.show()
    # plt.savefig('./venn.jpg')

    index = set(file.index) - set(no_dgsp | no_jones | no_p | low_p)
    return_file = file.loc[index, :].reset_index(drop=True)

    print('total dataset:', len(file.index))
    print('with resonable distance and parallax:', len(index))

    return return_file


def Mg_cut(data, mg_low=1, mg_up=7):
    cond_mg = ((data['qg_geo'] - data['ag_gspphot']) > 1) & ((data['qg_geo'] - data['ag_gspphot']) < 7)
    index = np.where(cond_mg)[0]
    return_data = data.loc[index, :].reset_index(drop=True)
    print('(Mg)0 in [%d,%d]: ' % (mg_low, mg_up), len(np.where(cond_mg)[0]))
    return return_data


# 信噪比筛选
# 复现 Weitao,Zhang: Fig2
def select_SNR(file, snrg_limit=10, snrr=False, snrr_limit=None):
    if snrr == False:
        snrr_limit = snrr_limit
    elif snrr == True:
        snrr_limit = snrg_limit * 10 / 7  # snrr / snrg = 10 / 7

    cond_snr_limit = (
                (file['snru'] > 0) & (file['snrg'] > 0) & (file['snrr'] > 0) & (file['snri'] > 0) & (file['snrz'] > 0)
                & (file['snru'] < 990) & (file['snrg'] < 990) & (file['snrr'] < 990) & (file['snri'] < 990) & (
                            file['snrz'] < 990))
    cond_high_snr = (file['snrg'] >= snrg_limit) & (file['snrr'] >= snrr_limit)

    all_index = set(np.where(cond_snr_limit)[0])
    high_snr_index = all_index & set(np.where(cond_high_snr)[0])
    low_snr_index = all_index - high_snr_index

    print('total sample:', len(file.index))
    print('low snr quality in:', len(low_snr_index))
    print('high snr quality in :', (len(high_snr_index)))

    high_snr = file.loc[high_snr_index, ['snrg', 'snrr']]
    low_snr = file.loc[low_snr_index, ['snrg', 'snrr']]

    return_file = file.loc[high_snr_index, :].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=200)
    ax.scatter(high_snr['snrg'], high_snr['snrr'], s=0.2, marker='.', color='black')
    ax.scatter(low_snr['snrg'], low_snr['snrr'], s=0.2, marker='.', color='grey')
    ax.set_xlabel('$S/N_{g}$')
    ax.set_ylabel('$S/N_{r}$')
    ax.set_xlim((0, 1000))
    ax.set_ylim((0, 1000))
    ax.set_xticks(np.arange(0, 1200, 200))
    ax.set_yticks(np.arange(0, 1200, 200))
    ax.plot([0, 10], [0, 10], transform=ax.transAxes, ls='--', c='r', label="$\\frac{S/N_{r}}{S/N_{g}}=\\frac{10}{7}$")
    ax.legend()
    return return_file


def select_fibermask(data):
    cond_fibermask = (data['fibermask'] == 0)
    index = np.where(cond_fibermask)[0]
    return_data = data.loc[index, :].reset_index(drop=True)
    print('origin number:', len(data.index))
    print('fibermask == 0:', len(np.where(cond_fibermask)[0]))
    return return_data


# 查看 bp_rp_0 与 mg_geo 是否完备
def check_color_mag_completeness(data):
    data['bp_rp_0'] = data['bp_rp'] - data['ebpminrp_gspphot']
    data['mg_geo'] = data['qg_geo'] - data['ag_gspphot']
    print('if there is no bp_rp_0:', np.where(np.isnan(data['bp_rp_0'])))
    print('if there is no mg_geo:', np.where(np.isnan(data['mg_geo'])))
    return data


#  bp_rp_0 与 mg_geo 完备的条件下，随机抽样 60000 条
def random_sample(data_dup, number=60000):
    # 注意！去除重复观测数据
    data = data_dup.drop_duplicates(subset=['uid'], keep='first').reset_index(drop=True)
    random.seed(42)  # 初始化随机数种子用以后续操作的重复实现
    rand_index = random.sample(set(np.array(data.index)), number)  # 随机取
    rand_sample = data.loc[rand_index, :].reset_index(drop=True)
    return rand_sample


# Fitting a continuous spectrum to the red and blue parts respectively
# find strong lines
def find_strong_line(wave, flux):
    # 用5次多项式拟合
    coeff = np.polyfit(wave, flux, 5)
    formula = np.poly1d(coeff)
    y = formula(wave)
    # 得到拟合线与强度的差值
    d = y - flux
    # 得到该差值的标准差
    sigma = np.std(d)
    # 当强度超出拟合线加减三倍标准差的范围时，认为是强线
    n = np.where((flux < y - 3 * sigma) | (flux > y + 3 * sigma))
    a = n[0]
    i = 0
    left = []
    right = []
    while i < len(a):
        head = i
        length = 0
        while (i < len(a) - 1) and (a[i + 1] == a[i] + 1):
            length += 1
            i += 1
        if head == 0:
            left.append(np.max([a[head] - length, 0]))
        else:
            left.append(np.max([a[head] - length, a[head - 1] + 1]))

        if i == len(a) - 1:
            right.append(np.min([a[i] + length, len(wave) - 1]))
        else:
            right.append(np.min([a[i] + length, a[i + 1] - 1]))
        i += 1
    return wave[left], wave[right]


# mask strong lines
def mask_strong_line(wave, flux, left, right):
    for i in range(len(left)):
        n = np.where((wave < left[i]) | (wave > right[i]))
        wave = wave[n]
        flux = flux[n]
    return [wave, flux]


# 迭代10次拟合连续谱
def find_conti_spec(wave, flux, nploy):
    for i in np.arange(10):
        coeff = np.polyfit(wave, flux, nploy)
        formula = np.poly1d(coeff)
        y = formula(wave)
        d = y - flux
        sigma = np.std(d)
        n = np.where(d < sigma)
        wave = wave[n]
        flux = flux[n]
    return formula


# 蓝端 5阶拟合
def normalize_blue_spec(wave, flux):
    left, right = find_strong_line(wave, flux)
    wave1, flux1 = mask_strong_line(wave, flux, left, right)
    formula = find_conti_spec(wave1, flux1, 5)
    conti = formula(wave)
    return flux / conti


# 红端 5阶拟合
def normalize_red_spec(wave, flux):
    left, right = find_strong_line(wave, flux)
    wave1, flux1 = mask_strong_line(wave, flux, left, right)
    formula = find_conti_spec(wave1, flux1, 5)
    conti = formula(wave)
    return flux / conti


# 找到文件夹下所有文件名
def get_fileNameList(file_dir):
    # root当前目录路径, dirs当前路径下所有子目录, files当前路径下所有非目录子文件
    for root, dirs, files in os.walk(file_dir):
        fileNameList = files
    return fileNameList


def flux_norm(error_file):
    wave = np.arange(3925, 8800, 1)

    hdu = fits.open(error_file, memmap=False)
    id = hdu[0].header['obsid']
    z = hdu[0].header['z']
    bad_pix = np.append(np.nonzero(hdu[1].data[0][3]), np.nonzero(hdu[1].data[0][4]))
    f_norm_fromfile = np.delete(hdu[1].data[0][5], bad_pix)  # 拉平光谱的流量
    f = np.delete(hdu[1].data[0][0], bad_pix)  # 去除坏像素，与标识位&或标识位
    w = np.delete(hdu[1].data[0][2], bad_pix)
    w = w / (1 + z)  # rest wave
    # !!!
    f_smooth = savgol_filter(f, 15, 7)
    # 分红蓝端去除连续谱
    index_b = np.where(w < 6000)
    index_r = np.where(w >= 6000)
    w_r = w[index_r]
    w_b = w[index_b]
    f_smooth_b = f_smooth[index_b]
    f_smooth_r = f_smooth[index_r]

    norm_b_smooth = normalize_blue_spec(w_b, f_smooth_b)
    norm_r_smooth = normalize_red_spec(w_r, f_smooth_r)
    norm_smooth = np.append(norm_b_smooth, norm_r_smooth)

    # 拟合
    f_inter_smooth = interp1d(w, norm_smooth)
    # 自己写的拉平函数的流量 插值统一波长范围
    f_normed_smooth = f_inter_smooth(wave)

    # !!!
    f_check = np.array(f_normed_smooth)
    cond = np.where((f_check > 1.2) | (f_check < 0.1))[0]

    if len(cond) != 0:
        f_check[cond] = 1

    '''
    w: 文件中包含的所有波长
    wave: (3925,8800)范围波长
    f: 原始流量
    f_smooth: 经过平滑的原始流量
    f_normed_smooth: 使用平滑后的原始流量的拉平光谱
    f_norm_fromfile: 文件中给出的拉平光谱
    f_check: 阈值检查后的拉平光谱
    '''
    return w, wave, f, f_smooth, f_normed_smooth, f_norm_fromfile, f_check
