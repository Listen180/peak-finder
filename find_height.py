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



# =============================================================================
# 
# =============================================================================
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


## 
img_save_path = './imgs/scatter_plot_all.png'
plt.figure()
plt.scatter(x, data["A"], color='green', label="A", s=5)
plt.scatter(x, data["B"], color='red', label="B", s=5)
plt.scatter(x, data["C"], color='purple', label="C", s=5)

plt.title(f'Scatter Plot', fontsize=16)
plt.xlabel('Default Index', fontsize=8)
plt.ylabel('Value', fontsize=8)
plt.legend()

plt.savefig(img_save_path, dpi=300)
print('')
print(f"Image saved to: \"{img_save_path}\"")
# plt.show()
plt.close()


# =============================================================================
# Fancy Process to Find the Support Line
# =============================================================================
def remove_above(y_org, curve, line):
    """
    """
    y_modified = y_org.copy()
    data_points_left = []
    for i, xv in enumerate(x):
        data_i = y_org[i]
        y_curve_i = curve[i]
        y_line_i = line[i]
        if y_curve_i < y_line_i:
            data_points_left.append(data_i)
    data_avg = np.mean(data_points_left)
    for i, xv in enumerate(x):
        data_i = y_org[i]
        y_curve_i = curve[i]
        y_line_i = line[i]
        if y_curve_i >= y_line_i:
            y_modified[i] = data_avg
    return y_modified



def remove_above_and_fit(y_org, y_fit, y_line, loop_num=2):
    """
    """
    for i in range(loop_num):
        temp = remove_above(y_org, y_fit, y_line)
        z_temp = np.polyfit(x, temp, 1, rcond=None, full=False, w=None, cov=False)
        p_temp = np.poly1d(z_temp)
        y_temp = p_temp(x)
        y_line = y_temp
    # print(p_temp)
    return y_temp


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
    
    peaks_index = peaks
    
    # =============================================================================
    # Poly Fit
    # =============================================================================
    deg = 10
    z = np.polyfit(x, y_org, deg, rcond=None, full=False, w=None, cov=False)
    
    p1 = np.poly1d(z)
    # print(p1)
    y_fit = p1(x)
    
    
    z2 = np.polyfit(x, y_org, 1, rcond=None, full=False, w=None, cov=False)
    p2 = np.poly1d(z2)
    # print(p2)
    y_line = p2(x)
    
    
    y_support = remove_above_and_fit(y_org, y_fit, y_line, loop_num=2)
    # =============================================================================
    # Plot
    # =============================================================================
    img_save_path = './imgs/poly_fit_'+col_name+'.png'
    plt.figure()
    plt.plot(x, y_org,'*',label='original values')
    plt.plot(x, y_fit,'r',label='polyfit values')
    plt.plot(x, y_line,'yellow',label='fit line')
    plt.plot(x, y_support,'grey',label='support line')
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
        
    return peak_height_org, peak_height_fit


if __name__ == "__main__":
    
    col_list = ["A", "B", "C"]
    for col in col_list:
        print('')
        print('')
        print(f"Processing on column {col} ... ")
        peak_height_org, peak_height_fit = analysis(data, col)



