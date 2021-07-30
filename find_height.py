#!/Users/leisen/anaconda3/bin/python3
# -*- coding: UTF-8 -*-

# ********************************************************
# * Author        : LEI Sen
# * Email         : sen.lei@outlook.com
# * Create time   : 2021-05-04 15:08
# * Last modified : 2021-05-04 23:54
# * Filename      : find_height.py
# * Description   : 
# *********************************************************


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import find_peaks
import math


# =============================================================================
# 
# =============================================================================



# =============================================================================
# Fancy Process to Find the Support Line
# =============================================================================
# def distance(p1, p2):
#     """
#     """
#     (x1, y1), (x2, y2) = p1, p2
#     dist = math.sqrt( (x1-x2)**2 + (y1-y2)**2 )
#     return dist



# def calc_area_by_side(a, b, c):
#     """
#     """
#     s = (a + b + c) / 2  
#     tri_area = math.sqrt( s * (s-a) * (s-b) * (s-c) )
#     return tri_area



def calc_area(p1, p2, p3):
    """
    """
    (x1, y1), (x2, y2), (x3, y3) = p1, p2, p3

    print(f"p1: {p1}, p2: {p2}, p3: {p3}")

    tri_area = 0.5 * abs(x2 * y3 + x1 * y2 + x3 * y1 - x3 * y2 - x2 * y1 - x1 * y3)

    return tri_area




# =============================================================================
# 
# =============================================================================
def analysis(data_org, col_name):
    """
    """
    y_org = data_org[col_name]
    # =============================================================================
    # Find Peak(s)
    # =============================================================================
    peaks, properties = find_peaks(y_org, width=10)
    gutter, properties_gutter = find_peaks(-y_org, width=2)
    
    img_save_path = './imgs/plot_with_peak_'+col_name+'.png'
    plt.figure()
    plt.plot(y_org)
    plt.plot(peaks, y_org[peaks], "x")
    plt.plot(np.zeros_like(y_org), "--", color="gray")
    
    plt.savefig(img_save_path, dpi=300)
    print('')
    print(f"Image saved to: \"{img_save_path}\"")
    # plt.show()
    plt.close()
    
    x_arr = np.array(x)

    peaks_index = peaks
    # peak_x = x[peaks_index[0]]
    peak_x = x_arr[peaks].mean()
    print(f"peak_x: {peak_x}")
    

    
    # =============================================================================
    # Poly Fit
    # =============================================================================
    deg = 10
    z = np.polyfit(x, y_org, deg, rcond=None, full=False, w=None, cov=False)
    
    p1 = np.poly1d(z)
    # print(f"p1: {p1}")
    y_fit = p1(x)
    
    
    z2 = np.polyfit(x, y_org, 1, rcond=None, full=False, w=None, cov=False)
    p2 = np.poly1d(z2)
    # print(f"p2: {p2}")
    y_line = p2(x)
    

    pp1 = np.polynomial.Polynomial.fit(x, y_org, deg)
    pp2 = np.polynomial.Polynomial.fit(x, y_org, 1)
    x_e = (pp1 - pp2).roots()
    

    x_e_gt_peak = x_e[(x_e > peak_x)]
    x_e_lt_peak = x_e[(x_e < peak_x)]

    # print("\n***"*3)
    # print(f"x_e_gt_peak: {x_e_gt_peak}")
    # print(f"x_e_lt_peak: {x_e_lt_peak}")
    # print("\n***"*3)

    x_left = x_e_lt_peak[-1]
    x_right = x_e_gt_peak[0]

    print(f"\nleft: {x_left}, peak: {peak_x}, right{x_right}")

    ## Get polynomial changing points
    # y = np.polyval(p1, x)
    Q = np.polyder(p1)  # f'
    xs = np.roots(Q)  # get the root of polynomial
    xs = xs[(xs>x[0]) & (xs<x[-1])]

    # print("\n********"*3)
    # print(f"  Diff 2nd: {y_d2}")
    # print(f"  xs: {xs}")


    Q2 = np.polyder(Q)  # f''
    y_d2 = np.polyval(Q2, xs)
    # is_gutter = [y_d2 > 0]
    is_gutter = np.array(y_d2 > 0)
    print(f"is_gutter: {is_gutter}")


    xs = np.array(xs)
    # xs = xs[tuple(is_gutter)]
    xs = xs[is_gutter]


    x_gutter_gt_peak = xs[(xs > peak_x)][0]
    x_gutter_lt_peak = xs[(xs < peak_x)][-1]

    xs = [x_gutter_lt_peak, x_gutter_gt_peak]
    ys = np.polyval(p1, xs)

    z_support = np.polyfit(xs, ys, 1, rcond=None, full=False, w=None, cov=False)
    p_support = np.poly1d(z_support)
    y_support = p_support(x)



    # =============================================================================
    # Plot
    # =============================================================================
    img_save_path = './imgs/poly_fit_'+col_name+'.png'
    plt.figure()
    plt.plot(x, y_org,'*',label='original values')
    plt.plot(x, y_fit,'r',label='polyfit values')
    plt.plot(x, y_line,'yellow',label='fit line')
    plt.plot(x, y_support,'grey',label='support line')
    plt.plot(xs, ys, "ro")
    plt.legend()
    
    plt.savefig(img_save_path, dpi=300)
    print('')
    print(f"Image saved to: \"{img_save_path}\"")
    # plt.show()
    plt.close()
    
    
    
    # =============================================================================
    # Calculate Peak Height
    # =============================================================================
    peaks_value = np.array(y_org[peaks])
    peaks_value_fit = np.array(y_fit[peaks])
    support_line_value = y_support[peaks_index]
    
    peak_height_org = peaks_value - support_line_value
    peak_height_fit = peaks_value_fit - support_line_value
    print("")
    print("----------------------------------------------")
    print(f"Peak Height (Orginal Data): {peak_height_org}")
    print(f"Peak Height (Curve Fitted Data): {peak_height_fit}")
    print("----------------------------------------------")

    cor1 = [xs[0], ys[0]]
    cor2 = [xs[1], ys[1]]
    cor3 = [x[peaks[0]], y_org[peaks[0]]]

    # a = distance(cor1, cor2)
    # b = distance(cor1, cor3)
    # c = distance(cor2, cor3)
    peak_area = calc_area(cor1, cor2, cor3)

    print("")
    print("----------------------------------------------")
    print(f"Peak Area: {peak_area}")
    print("----------------------------------------------")
        
    return peak_height_org, peak_height_fit, peak_area



if __name__ == "__main__":
    

    input_f = "./data/curve.txt"

    data = pd.read_csv(input_f, header=None, sep="\t")
    data.columns = ["A", "B", "C"]

    y1 = data.loc[:, "A"]
    y2 = data.loc[:, "B"]
    y3 = data.loc[:, "C"]

    x_start = -600
    x_end = 100
    x_range = range(x_start, x_end, int((x_end-x_start)/len(data)))
    x = list(x_range)
    x = x[0:len(data)]


    col_list = ["A", "B", "C"]
    for col in col_list:
        print('')
        print('')
        print(f"Processing on column {col} ... ")
        peak_height_org, peak_height_fit, peak_area = analysis(data, col)



