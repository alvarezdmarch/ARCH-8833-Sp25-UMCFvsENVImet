import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import math
import os

file_path = r"your/file/path/here.xlsx"  # Replace with your file path

output_folder = r"your/output/folder/path"  # Replace with your output folder path

df = pd.read_excel(file_path)
#print(df.head())

def calculate_statistics(base_series, model_series, variable_name="Variable"):
    valid_idx = base_series.notna() & model_series.notna()

    base = base_series[valid_idx]
    model = model_series[valid_idx]

    n = len(base)

    diff = model - base
    abs_diff = diff.abs()

    stats_results = {}
    stats_results['Variable'] = variable_name
    stats_results['N_Points'] = n
    stats_results['MBE'] = diff.mean()
    stats_results['MAE'] = abs_diff.mean()
    stats_results['MSE'] = (diff**2).mean()
    stats_results['RMSE'] = math.sqrt(stats_results['MSE'])

    mean_base = base.mean()
    mean_sum = (model + base).mean()

    stats_results['NMAE'] = stats_results['MAE'] / mean_base if mean_base != 0 else np.nan
    stats_results['NRMSE (CVRMSE)'] = stats_results['RMSE'] / mean_base if mean_base != 0 else np.nan

    #Pearson Correlation
    if base.std() == 0 or model.std() == 0:
        corr, p_value = (np.nan, np.nan)
    else:
        corr, p_value = stats.pearsonr(base, model)

    stats_results['Pearson_r'] = corr
    stats_results['R_squared'] = corr**2 if not np.isnan(corr) else np.nan
    stats_results['p_value'] = p_value

    #Index of Agreement (d)
    num_d = (diff**2).sum()
    term1 = (model - mean_base).abs()
    term2 = (base - mean_base).abs()
    den_d_series = (term1 + term2)**2
    den_d_sum = den_d_series.sum()

    stats_results['Index_of_Agreement_d'] = 1 - (num_d / den_d_sum) if den_d_sum != 0 else np.nan

    #Fractional Bias (FB)
    stats_results['FB'] = 2 * diff.mean() / mean_sum if mean_sum != 0 else np.nan

    #FAC2
    ratio = model.divide(base.replace(0, np.nan))
    fac2_count = ((ratio >= 0.5) & (ratio <= 2.0)).sum()
    valid_fac2_points = ratio.notna().sum()
    stats_results['FAC2'] = fac2_count / valid_fac2_points if valid_fac2_points > 0 else np.nan

    # Store the column names for reference
    stats_results['base_col'] = base_series.name
    stats_results['model_col'] = model_series.name
    
    return stats_results

variables_to_analyze = [
    (df.columns[0], df.columns[1], "Temperature at 1.7m in Domain Box"),
    (df.columns[2], df.columns[3], "Humidity at 1.7m in Domain Box"),
    (df.columns[4], df.columns[5], "Wind Speed at 1.7m in Domain Box"),
    (df.columns[6], df.columns[7], "Temperature at 1.7m in Refinement Box"),
    (df.columns[8], df.columns[9], "Humidity at 1.7m in Refinement Box"),
    (df.columns[10], df.columns[11], "Wind Speed at 1.7m in Refinement Box"),
]

all_stats = []

for base_col, model_col, var_name in variables_to_analyze:
    stats_dict = calculate_statistics(df[base_col], df[model_col], variable_name=var_name)
    all_stats.append(stats_dict)

if all_stats:
    stats_df = pd.DataFrame(all_stats)
    stats_df = stats_df.round(4)

    stats_output_file = r"your/output/folder/path/stats_output.xlsx"
    stats_df.to_excel(stats_output_file, index=False)

def plot_scatter(base_series, model_series, stats_dict, title="Plot"):
    plt.figure(figsize=(10, 8))

    valid_idx = base_series.notna() & model_series.notna()
    base = base_series[valid_idx]
    model = model_series[valid_idx]

    min_val = min(base.min(), model.min()) * 0.95
    max_val = max(base.max(), model.max()) * 1.05

    if abs(max_val - min_val) < 1e-6:
        min_val -= 0.5
        max_val += 0.5

    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1 Line', zorder=1) #1:1 line
    plt.scatter(base, model, color="grey", alpha=1.0, label='Data Points', zorder=2, s=5)

    base_col_name = stats_dict.get('base_col', base_series.name)
    model_col_name = stats_dict.get('model_col', model_series.name)
    plt.xlabel(f"ENVI-met ({base_col_name})")
    plt.ylabel(f"UMCF ({model_col_name})")
    plt.title(f"{title} ({stats_dict['Variable']})")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.gca().set_aspect('equal', adjustable='box')

    stats_text = (f"R² = {stats_dict['R_squared']:.3f}\n"
                  f"RMSE = {stats_dict['RMSE']:.3f}\n"
                  f"MBE = {stats_dict['MBE']:.3f}\n"
                  f"d = {stats_dict['Index_of_Agreement_d']:.3f}\n"
                  f"N = {stats_dict['N_Points']}")
    plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes,
             fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))
    #plt.legend(loc='lower right')
    plt.tight_layout()

    try:
        plot_filename = f"scatter_{stats_dict['Variable'].replace(' ', '_').replace('/', '_')}.png"
        plot_filepath = os.path.join(output_folder, plot_filename)
        plt.savefig(plot_filepath)
        print(f"Scatter plot saved to '{plot_filepath}'")
    except Exception as e:
        print(f"Error saving plot '{title}': {e}")
    plt.show()

def plot_residuals(base_series, model_series, stats_dict, title="Residual Plot"):
    plt.figure(figsize=(10, 6))

    valid_idx = base_series.notna() & model_series.notna()
    base = base_series[valid_idx]
    residuals = (model_series - base_series)[valid_idx]

    plt.scatter(base, residuals, color="grey", alpha=1.0, s=5)
    plt.axhline(0, color='red', linestyle='--', linewidth=1)
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.xlabel(f"ENVI-met ({base_series.name})")
    plt.ylabel("Residuals (UMCF - ENVI-met)")
    plt.title(f"{title} ({stats_dict['Variable']})")
    plt.tight_layout()

    try:
        plot_filename = f"residual_{stats_dict['Variable'].replace(' ', '_').replace('/', '_')}.png"
        plot_filepath = os.path.join(output_folder, plot_filename)
        plt.savefig(plot_filepath)
        print(f"Residual plot saved to '{plot_filepath}'")
    except Exception as e:
        print(f"Error saving residual plot '{title}': {e}")
    plt.show()

def plot_difference_histogram(base_series, model_series, stats_dict, title="Difference Histogram"):
    valid_idx = base_series.notna() & model_series.notna()
    differences = (model_series - base_series)[valid_idx]

    plt.figure(figsize=(10, 6))
    plt.hist(differences, bins=30, color='grey', alpha=1.0)
    plt.axvline(differences.mean(), color='red', linestyle='--', label=f"Mean = {differences.mean():.3f}")
    plt.axvline(0, color='black', linestyle='-', linewidth=1)
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.xlabel("Difference (UMCF - ENVI-met)")
    plt.ylabel("Frequency")
    plt.title(f"{title} ({stats_dict['Variable']})")
    plt.legend()
    plt.tight_layout()

    try:
        plot_filename = f"difference_histogram_{stats_dict['Variable'].replace(' ', '_').replace('/', '_')}.png"
        plot_filepath = os.path.join(output_folder, plot_filename)
        plt.savefig(plot_filepath)
        print(f"Difference histogram saved to '{plot_filepath}'")
    except Exception as e:
        print(f"Error saving histogram '{title}': {e}")
    plt.show()

def plot_boxplot(errors_df):
    errors_df.boxplot(figsize=(12, 6), grid=False, color='grey', patch_artist=True)
    plt.title("Boxplot of Errors Across Variables")
    plt.ylabel("Difference (UMCF - ENVI-met)", fontsize=10)
    plt.xlabel("ENVI-met T", fontsize=10)
   
    plt.xticks(rotation=30, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.axhline(y=0, color='r', linestyle='-', linewidth=1, alpha=0.7)

    for i, column in enumerate(errors_df.columns):
        mean_val = errors_df[column].mean()
        plt.text(i+1, mean_val, f'μ={mean_val:.2f}', 
                 ha='center', va='bottom', fontsize=10, 
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2'))

    plot_filepath = os.path.join(output_folder, "error_boxplot.png")
    plt.savefig(plot_filepath)
    print(f"Boxplot saved to '{plot_filepath}'")

'''
if "stats_df" in locals():
    for index, row in stats_df.iterrows():
        base_col = row['base_col']
        model_col = row['model_col']
        
        if base_col in df.columns and model_col in df.columns:
            plot_scatter(df[base_col], df[model_col], row.to_dict(), title="ENVI-met vs UMCF")
        else:
            print(f"Warning: Could not find columns {base_col} and/or {model_col} in the dataframe")
'''

if "stats_df" in locals():
    differences_df = pd.DataFrame()  #To collect differences for boxplot
    
    for index, row in stats_df.iterrows():
        base_col = row['base_col']
        model_col = row['model_col']
        var_name = row['Variable']
        
        if base_col in df.columns and model_col in df.columns:
            #Create all individual plots
            plot_scatter(df[base_col], df[model_col], row.to_dict(), title="ENVI-met vs UMCF")
            plot_residuals(df[base_col], df[model_col], row.to_dict(), title="Residuals")
            plot_difference_histogram(df[base_col], df[model_col], row.to_dict(), title="Difference Distribution")
            
            #Collect differences for boxplot
            valid_idx = df[base_col].notna() & df[model_col].notna()
            diff = df[model_col][valid_idx] - df[base_col][valid_idx]
            differences_df[var_name] = diff
            
        else:
            print(f"Warning: Could not find columns {base_col} and/or {model_col} in the dataframe")
    
    if not differences_df.empty:
        plot_boxplot(differences_df)
    else:
        print("No valid data available for boxplot creation")