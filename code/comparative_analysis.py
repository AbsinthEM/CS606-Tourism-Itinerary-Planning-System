"""
Comparative Analysis for Tourism Itinerary Planning
Processes results from Stage 1 (initial planning) and Stage 2 (disruption handling)
to generate comprehensive analysis and visualizations.
"""

import os
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# --- Helper Functions for Parsing Result Files ---
def parse_stage1_summary(file_path):
    """
    Parse a Stage 1 summary file and extract relevant metrics.

    Returns:
        dict: Dictionary containing extracted metrics
    """
    metrics = {}

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Extract tourist ID
    match = re.search(r'Tourist ID: (\d+)', content)
    if match:
        metrics['tourist_id'] = int(match.group(1))

    # Extract preferences
    match = re.search(r'Preferences: \[(.*?)\]', content)
    if match:
        metrics['preferences'] = match.group(1).split(', ')

    # Extract objective values
    match = re.search(r'Initial Solution: ([\d\.]+)', content)
    if match:
        metrics['initial_objective'] = float(match.group(1))

    match = re.search(r'Optimized Solution: ([\d\.]+)', content)
    if match:
        metrics['optimized_objective'] = float(match.group(1))

    # Compute improvement from numeric columns:
    if 'initial_objective' in metrics and 'optimized_objective' in metrics:
        init_val = metrics['initial_objective']
        opt_val = metrics['optimized_objective']
        if init_val != 0:
            derived_improvement = ((opt_val - init_val) / abs(init_val)) * 100
            metrics['improvement_percentage'] = derived_improvement
        

    # Extract attraction counts
    match = re.search(r'Attractions Count\s+(\d+)\s+(\d+)', content)
    if match:
        metrics['initial_attractions'] = int(match.group(1))
        metrics['optimized_attractions'] = int(match.group(2))

    # Extract money spent
    match = re.search(r'Money Spent \(\$\)\s+([\d\.]+)\s+([\d\.]+)', content)
    if match:
        metrics['initial_money_spent'] = float(match.group(1))
        metrics['optimized_money_spent'] = float(match.group(2))

    return metrics

def parse_stage2_comparison(file_path):
    """
    Parse a Stage 2 comparison file and extract relevant metrics.

    Returns:
        dict: Dictionary containing extracted metrics
    """
    metrics = {}

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Extract tourist ID
    match = re.search(r'Tourist ID: (\d+)', content)
    if match:
        metrics['tourist_id'] = int(match.group(1))

    # Extract method averages
    methods = ['Initial', 'Optimized', 'Quick Repair', 'ALNS Replanning']
    method_keys = ['initial', 'optimized', 'quick_repair', 'alns_replan']

    # Extract average objectives for each method from the overall average values section
    average_section = re.search(r'OVERALL AVERAGE OBJECTIVE VALUES.*?-+\s+(.*?)(?:\n\n|\Z)', content, re.DOTALL)
    if average_section:
        avg_content = average_section.group(1)
        for method, key in zip(methods, method_keys):
            # match = re.search(rf'{method}\s+([\d\.]+)\s+', avg_content, re.IGNORECASE)
            # match = re.search(rf'{method}\s+([-]?\d+(?:\.\d+)?)\s+', avg_content, re.IGNORECASE)
            match = re.search(rf'{method}\s+([-]?\d+(?:\.\d+)?)\s+([-]?\d+(?:\.\d+)?)', avg_content, re.IGNORECASE)
            if match:
                metrics[f'{key}_avg_objective'] = float(match.group(1))

        # Extract average attractions
        for method, key in zip(methods, method_keys):
            # match = re.search(rf'{method}\s+[\d\.]+\s+([\d\.]+)', avg_content, re.IGNORECASE)
            # match = re.search(rf'{method}\s+[\d\.]+\s+([\d\.]+)', avg_content, re.IGNORECASE)
            match = re.search(rf'{method}\s+([-]?\d+(?:\.\d+)?)\s+([-]?\d+(?:\.\d+)?)', avg_content, re.IGNORECASE)
            if match:
                metrics[f'{key}_avg_attractions'] = float(match.group(1))

    # Extract stability and proximity scores for ALNS replanning (if available)
    scenarios_section = re.search(r'Method: ALNS Replanning\n.*?Scenario\s+Objective\s+Stability\s+Proximity\s+(.*?)(?:\n\n|\Z)', content, re.DOTALL)
    if scenarios_section:
        scenarios_text = scenarios_section.group(1)
        # stability_values = re.findall(r'\d+\s+[\d\.]+\s+([\d\.]+)\s+', scenarios_text)
        # proximity_values = re.findall(r'\d+\s+[\d\.]+\s+[\d\.]+\s+([\d\.]+)', scenarios_text)
        
        # Updated patterns (allow negative objectives):
        stability_values = re.findall(
            r'\d+\s+[-]?\d+(?:\.\d+)?\s+([\d\.]+)\s+', 
            scenarios_text
        )
        proximity_values = re.findall(
            r'\d+\s+[-]?\d+(?:\.\d+)?\s+[\d\.]+\s+([\d\.]+)', 
            scenarios_text
        )

        if stability_values:
            metrics['alns_avg_stability'] = np.mean([float(v) for v in stability_values])

        if proximity_values:
            metrics['alns_avg_proximity'] = np.mean([float(v) for v in proximity_values])

    return metrics

# --- Analysis Functions ---
def analyze_stage1_results(directory="stage1_results"):
    """
    Analyze all Stage 1 results and return a DataFrame with metrics.

    Args:
        directory (str): Directory containing Stage 1 result files

    Returns:
        pd.DataFrame: DataFrame with Stage 1 metrics for all tourists
    """
    results = []

    # Find all Stage 1 summary files
    if os.path.exists(directory):
        for file in os.listdir(directory):
            if file.endswith("_Summary.txt") and "Tourist_" in file:
                file_path = os.path.join(directory, file)

                try:
                    metrics = parse_stage1_summary(file_path)
                    results.append(metrics)
                except Exception as e:
                    print(f"Error parsing {file}: {e}")

    # Convert to DataFrame
    if results:
        df = pd.DataFrame(results)
        if 'tourist_id' in df.columns:
            df.set_index('tourist_id', inplace=True)
        return df
    else:
        print(f"No valid Stage 1 results found in {directory}")
        return pd.DataFrame()

def analyze_stage2_results(directory="stage2_results"):
    """
    Analyze all Stage 2 results and return a DataFrame with metrics.

    Args:
        directory (str): Directory containing Stage 2 result files

    Returns:
        pd.DataFrame: DataFrame with Stage 2 metrics for all tourists
    """
    results = []

    # Find all Stage 2 comparison files
    if os.path.exists(directory):
        for file in os.listdir(directory):
            if file.endswith("_disruption_comparison.txt") and "tourist_" in file:
                file_path = os.path.join(directory, file)

                try:
                    metrics = parse_stage2_comparison(file_path)
                    results.append(metrics)
                except Exception as e:
                    print(f"Error parsing {file}: {e}")

    # Convert to DataFrame
    if results:
        df = pd.DataFrame(results)
        if 'tourist_id' in df.columns:
            df.set_index('tourist_id', inplace=True)
        return df
    else:
        print(f"No valid Stage 2 results found in {directory}")
        return pd.DataFrame()

def merge_stage_results(stage1_df, stage2_df):
    """
    Merge Stage 1 and Stage 2 results into a single DataFrame.

    Args:
        stage1_df (pd.DataFrame): Stage 1 results DataFrame
        stage2_df (pd.DataFrame): Stage 2 results DataFrame

    Returns:
        pd.DataFrame: Combined DataFrame with results from both stages
    """
    # If either DataFrame is empty, return the other
    if stage1_df.empty:
        return stage2_df
    if stage2_df.empty:
        return stage1_df

    # Merge DataFrames on tourist_id index
    combined_df = pd.merge(stage1_df, stage2_df, left_index=True, right_index=True, how='outer')

    # Fill missing values with NaN rather than 0
    # This distinguishes between "no data" and "zero value"

    return combined_df


def generate_summary_statistics(df, output_file=None):
    """
    Generate summary statistics and save to a text file.

    Args:
        df (pd.DataFrame): DataFrame with merged results
        output_file (str, optional): Path to save the output file

    Returns:
        str: Summary statistics text
    """
    lines = []
    lines.append("TOURISM ITINERARY PLANNING - COMPREHENSIVE ANALYSIS")
    lines.append("=" * 70)
    lines.append("")

    # Number of tourists analyzed
    num_tourists = len(df)
    lines.append(f"Total tourists analyzed: {num_tourists}")

    # Average objectives by method
    lines.append("\nAVERAGE OBJECTIVE VALUES BY METHOD:")
    lines.append("-" * 50)

    methods = [
        ('initial_objective', 'Initial Heuristic'),
        ('optimized_objective', 'ALNS Optimized'),
        ('quick_repair_avg_objective', 'Quick Repair (with disruptions)'),
        ('alns_replan_avg_objective', 'ALNS Replanning (with disruptions)')
    ]

    for col, name in methods:
        if col in df.columns and not df[col].empty:
            avg_value = df[col].mean()
            lines.append(f"{name}: {avg_value:.4f}")

    # Median calculation
    lines.append("\nMEDIAN OBJECTIVE VALUES BY METHOD:")
    lines.append("-" * 50)
    for col, name in methods:
        if col in df.columns and not df[col].empty:
            median_value = df[col].median()
            lines.append(f"{name} median: {median_value:.4f}")
            
    # Count cases where ALNS doesn't improve
    lines.append("\nCOUNTS OF CASES WHERE ALNS DIDN'T IMPROVE:")
    lines.append("-" * 50)
    
    # Count cases for Stage 1
    stage1_count = 0
    if 'improvement_percentage' in df.columns and not df['improvement_percentage'].empty:
        stage1_count = df[df['improvement_percentage'] == 0].shape[0]
    
    # Count cases for Stage 2
    stage2_count = 0
    if ('quick_repair_avg_objective' in df.columns and 'alns_replan_avg_objective' in df.columns):
        stage2_count = (df['quick_repair_avg_objective'] > df['alns_replan_avg_objective']).sum()
    
    lines.append(f"Stage 1: {stage1_count} ({stage1_count/len(df)*100:.1f}%)")
    lines.append(f"Stage 2: {stage2_count} ({stage2_count/len(df)*100:.1f}%)")
    


    # ------- STAGE 1 ANALYSIS: ALNS OPTIMIZATION -------
    
    lines.append("\nSTAGE 1: ALNS OPTIMIZATION ANALYSIS")
    lines.append("=" * 50)
    
    # Average improvement from initial to optimized
    if 'improvement_percentage' in df.columns and not df['improvement_percentage'].empty:
        avg_improvement = df['improvement_percentage'].mean()
        median_improvement = df['improvement_percentage'].median()

        lines.append(f"Average improvement from Initial to Optimized: {avg_improvement:.2f}%")
        lines.append(f"Median improvement from Initial to Optimized: {median_improvement:.2f}%")
        
        zero_improvement_count = df[df['improvement_percentage'] == 0].shape[0]
        
    
        # Count cases where initial and optimized objectives are identical (zero improvement)
        zero_improvement_count = df[df['improvement_percentage'] == 0].shape[0]
        safety_net_percentage = (zero_improvement_count / len(df)) * 100
        
        lines.append(f"\nSAFETY NET ACTIVATION ANALYSIS (STAGE 1):")
        lines.append("-" * 50)
        lines.append(f"Cases where ALNS reverted to initial solution: {zero_improvement_count} ({safety_net_percentage:.1f}%)")
        lines.append("Note: Stage 1 has an existing safety net implementation that reverts to the initial")
        lines.append("solution when ALNS fails to find an improvement.")
        
        # Break down by tourist preference groups - handling lists properly
        if 'preferences' in df.columns:
            try:
                # Convert preferences to string representation for grouping
                df_zero = df[df['improvement_percentage'] == 0].copy()
                df_zero['pref_str'] = df_zero['preferences'].apply(lambda x: str(x))
                pref_groups = df_zero.groupby('pref_str').size()
                
                if len(pref_groups) > 0:
                    lines.append("\nBreakdown by preference group:")
                    for pref, count in pref_groups.items():
                        lines.append(f"  {pref}: {count} cases")
            except Exception as e:
                lines.append(f"\nUnable to group by preferences: {str(e)}")
    

    # ------- STAGE 2 ANALYSIS: DISRUPTION HANDLING -------
    
    lines.append("\nSTAGE 2: DISRUPTION HANDLING ANALYSIS")
    lines.append("=" * 50)
    
    # Derive Stage 2 improvement row-by-row if Quick Repair & ALNS columns exist
    if 'quick_repair_avg_objective' in df.columns and 'alns_replan_avg_objective' in df.columns:
        df_stage2 = df.dropna(subset=['quick_repair_avg_objective', 'alns_replan_avg_objective']).copy()
        
        df_stage2['stage2_improvement_pct'] = (
            (df_stage2['alns_replan_avg_objective'] - df_stage2['quick_repair_avg_objective'])
            / df_stage2['quick_repair_avg_objective'].abs()
        ) * 100
        
        if not df_stage2['stage2_improvement_pct'].empty:
            stage2_mean = df_stage2['stage2_improvement_pct'].mean()
            stage2_median = df_stage2['stage2_improvement_pct'].median()

            lines.append(f"Mean improvement from Quick Repair to ALNS Replan: {stage2_mean:.2f}%")
            lines.append(f"Median improvement from Quick Repair to ALNS Replan: {stage2_median:.2f}%")
        else:
            lines.append("No valid Stage 2 improvement data found (missing columns or values).")
    
    
    # ALNS replanning specific metrics
    if 'alns_avg_stability' in df.columns and not df['alns_avg_stability'].empty:
        lines.append("ALNS REPLANNING METRICS:")
        lines.append("-" * 50)

        # Mean values
        avg_stability = df['alns_avg_stability'].mean()
        lines.append(f"Mean stability score: {avg_stability:.4f}")

        if 'alns_avg_proximity' in df.columns and not df['alns_avg_proximity'].empty:
            avg_proximity = df['alns_avg_proximity'].mean()
            lines.append(f"Mean proximity score: {avg_proximity:.4f}")
        
        # Median values - make sure we calculate and include these
        if not df['alns_avg_stability'].empty:
            median_stability = df['alns_avg_stability'].median()
            lines.append(f"Median stability score: {median_stability:.4f}")
        
        if 'alns_avg_proximity' in df.columns and not df['alns_avg_proximity'].empty:
            median_proximity = df['alns_avg_proximity'].median()
            lines.append(f"Median proximity score: {median_proximity:.4f}")

    # Method comparison for disruption handling
    if ('quick_repair_avg_objective' in df.columns and
        'alns_replan_avg_objective' in df.columns and
        not df['quick_repair_avg_objective'].empty and
        not df['alns_replan_avg_objective'].empty):

        lines.append("\nDISRUPTION HANDLING METHOD COMPARISON:")
        lines.append("-" * 50)

        qr_avg = df['quick_repair_avg_objective'].mean()
        alns_avg = df['alns_replan_avg_objective'].mean()
        qr_median = df['quick_repair_avg_objective'].median()
        alns_median = df['alns_replan_avg_objective'].median()

        if qr_avg > 0 and alns_avg > 0:
            diff_pct = (alns_avg - qr_avg) / qr_avg * 100
            median_diff_pct = (alns_median - qr_median) / qr_median * 100
            better_avg = "ALNS Replanning" if diff_pct > 0 else "Quick Repair"
            better_median = "ALNS Replanning" if median_diff_pct > 0 else "Quick Repair"
            
            lines.append(f"{better_avg} achieves {abs(diff_pct):.2f}% better mean objective values on average")
            lines.append(f"{better_median} achieves {abs(median_diff_pct):.2f}% better median objective values")

        # Count how many times each method was better
        method_wins = {'quick_repair': 0, 'alns_replan': 0, 'tie': 0}

        for idx, row in df.iterrows():
            qr_obj = row.get('quick_repair_avg_objective')
            alns_obj = row.get('alns_replan_avg_objective')

            if not (np.isnan(qr_obj) or np.isnan(alns_obj)):
                if qr_obj > alns_obj:
                    method_wins['quick_repair'] += 1
                elif alns_obj > qr_obj:
                    method_wins['alns_replan'] += 1
                else:
                    method_wins['tie'] += 1

        lines.append(f"Quick Repair was better for {method_wins['quick_repair']} tourists")
        lines.append(f"ALNS Replanning was better for {method_wins['alns_replan']} tourists")
        lines.append(f"Methods tied for {method_wins['tie']} tourists")

        # Calculate win percentage
        valid_comparisons = sum(method_wins.values())
        if valid_comparisons > 0:
            qr_win_pct = method_wins['quick_repair'] / valid_comparisons * 100
            alns_win_pct = method_wins['alns_replan'] / valid_comparisons * 100

            lines.append(f"Quick Repair win rate: {qr_win_pct:.1f}%")
            lines.append(f"ALNS Replanning win rate: {alns_win_pct:.1f}%")
            
        # Safety net and outlier analysis for Stage 2
        # First, calculate the difference between Quick Repair and ALNS Replanning
        df['obj_diff'] = df['quick_repair_avg_objective'] - df['alns_replan_avg_objective']
        
        # Identify cases where Quick Repair outperformed ALNS Replanning
        qr_win_mask = df['obj_diff'] > 0
        qr_outperform_count = qr_win_mask.sum()
        
        # Calculate the average margin by which QR outperformed ALNS
        if qr_outperform_count > 0:
            qr_margin = df.loc[qr_win_mask, 'obj_diff'].mean()
            qr_median_margin = df.loc[qr_win_mask, 'obj_diff'].median()
            
            # Find top 5 cases with biggest margins
            top_outliers = df[qr_win_mask].sort_values(by='obj_diff', ascending=False).head(5)
            
            lines.append("\nSTAGE 2 SAFETY NET ANALYSIS (FUTURE IMPLEMENTATION):")
            lines.append("-" * 50)
            lines.append(f"Cases where Quick Repair outperformed ALNS Replanning: {qr_outperform_count} ({qr_win_pct:.1f}%)")
            lines.append(f"Average margin in these cases: {qr_margin:.4f}")
            lines.append(f"Median margin in these cases: {qr_median_margin:.4f}")
            lines.append("Note: Unlike Stage 1, Stage 2 currently has no safety net implementation.")
            
            lines.append("\nTop 5 cases where Quick Repair outperforms ALNS Replanning:")
            lines.append("-" * 70)
            for tourist_id, row in top_outliers.iterrows():
                qr_obj = row['quick_repair_avg_objective']
                alns_obj = row['alns_replan_avg_objective']
                difference = row['obj_diff']
                
                # Safely convert preferences to string
                try:
                    preferences = str(row.get('preferences', 'Unknown'))
                except:
                    preferences = "Unknown"
                
                lines.append(f"Tourist {tourist_id}: QR={qr_obj:.4f}, ALNS={alns_obj:.4f}, Diff={difference:.4f}")
                lines.append(f"  Preferences: {preferences}")
                
                if 'initial_attractions' in row and 'optimized_attractions' in row:
                    lines.append(f"  Attractions: Initial={row['initial_attractions']}, Optimized={row['optimized_attractions']}")
            
            # Estimate improvement with safety net using both mean and median
            potential_alns_avg = df['alns_replan_avg_objective'].copy()
            potential_alns_avg[qr_win_mask] = df.loc[qr_win_mask, 'quick_repair_avg_objective']
            
            current_alns_avg = df['alns_replan_avg_objective'].mean()
            improved_alns_avg = potential_alns_avg.mean()
            improvement_pct = ((improved_alns_avg - current_alns_avg) / current_alns_avg) * 100
            
            current_alns_median = df['alns_replan_avg_objective'].median()
            improved_alns_median = potential_alns_avg.median()
            median_improvement_pct = ((improved_alns_median - current_alns_median) / current_alns_median) * 100
            
            lines.append("\nPotential Safety Net Implementation for Stage 2:")
            lines.append(f"Current ALNS Replanning mean: {current_alns_avg:.4f}")
            lines.append(f"With Safety Net mean: {improved_alns_avg:.4f}")
            lines.append(f"Mean improvement: {improvement_pct:.2f}%")
            
            lines.append(f"Current ALNS Replanning median: {current_alns_median:.4f}")
            lines.append(f"With Safety Net median: {improved_alns_median:.4f}")
            lines.append(f"Median improvement: {median_improvement_pct:.2f}%")
            
            lines.append("Recommendation: Implement a safety net in Stage 2 that selects the better")
            lines.append("solution between Quick Repair and ALNS Replanning for each tourist.")

    summary_text = "\n".join(lines)

    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(summary_text)

    return summary_text



def export_non_improving_cases(df, output_dir="analysis_results"):
    """
    Identify cases where ALNS didn't improve over heuristics in Stage 1 and Stage 2,
    and export them to separate CSV files for further analysis.
    
    Args:
        df (pd.DataFrame): DataFrame with merged results
        output_dir (str): Directory to save output files
    """
    print("Exporting cases where ALNS didn't improve...")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # STAGE 1: Cases where ALNS didn't improve over initial solution
    if 'improvement_percentage' in df.columns:
        stage1_cases = df[df['improvement_percentage'] == 0].copy()
        
        if not stage1_cases.empty:
            # Select relevant columns for analysis
            cols_to_keep = ['initial_objective', 'optimized_objective', 'improvement_percentage', 
                           'initial_attractions', 'optimized_attractions', 'preferences']
            
            # Add any other columns that exist in the dataframe
            extra_cols = ['initial_money_spent', 'optimized_money_spent']
            for col in extra_cols:
                if col in df.columns:
                    cols_to_keep.append(col)
            
            # Keep only columns that exist in the dataframe
            cols_to_keep = [col for col in cols_to_keep if col in stage1_cases.columns]
            
            # Export to CSV with tourist_id as index
            stage1_output = os.path.join(output_dir, "stage1_non_improving_cases.csv")
            stage1_cases[cols_to_keep].to_csv(stage1_output)
            print(f"Exported {len(stage1_cases)} Stage 1 non-improving cases to {stage1_output}")
    
    # STAGE 2: Cases where ALNS Replanning performed worse than Quick Repair
    if 'quick_repair_avg_objective' in df.columns and 'alns_replan_avg_objective' in df.columns:
        # Calculate difference between Quick Repair and ALNS Replanning
        df['obj_diff'] = df['quick_repair_avg_objective'] - df['alns_replan_avg_objective']
        
        # Cases where Quick Repair outperformed ALNS Replanning
        stage2_cases = df[df['obj_diff'] > 0].copy()
        
        if not stage2_cases.empty:
            # Sort by difference (largest first)
            stage2_cases = stage2_cases.sort_values(by='obj_diff', ascending=False)
            
            # Select relevant columns for analysis
            cols_to_keep = ['quick_repair_avg_objective', 'alns_replan_avg_objective', 'obj_diff',
                           'quick_repair_avg_attractions', 'alns_replan_avg_attractions', 
                           'preferences']
            
            # Add stability and proximity if available
            extra_cols = ['alns_avg_stability', 'alns_avg_proximity']
            for col in extra_cols:
                if col in df.columns:
                    cols_to_keep.append(col)
            
            # Keep only columns that exist in the dataframe
            cols_to_keep = [col for col in cols_to_keep if col in stage2_cases.columns]
            
            # Export to CSV with tourist_id as index
            stage2_output = os.path.join(output_dir, "stage2_non_improving_cases.csv")
            stage2_cases[cols_to_keep].to_csv(stage2_output)
            print(f"Exported {len(stage2_cases)} Stage 2 non-improving cases to {stage2_output}")

def plot_stages_win_comparison(df, output_dir="analysis_results"):
    """
    Create pie charts showing which method performs better across tourists for both Stage 1 and Stage 2.
    
    Args:
        df (pd.DataFrame): DataFrame with merged results
        output_dir (str): Directory to save output files
    """
    plt.figure(figsize=(15, 6))

    # Stage 1: Improvement from Initial to Optimized
    plt.subplot(1, 2, 1)
    if 'improvement_percentage' in df.columns:
        # Categorize improvement
        improvement_categories = pd.cut(df['improvement_percentage'], 
                                        bins=[-float('inf'), 0, float('inf')], 
                                        labels=['Initial Heuristic Wins', 'ALNS Wins'])
        
        stage1_wins = improvement_categories.value_counts()
        
        if not stage1_wins.empty:
            plt.pie(stage1_wins, labels=stage1_wins.index, autopct='%1.1f%%', 
                    colors=['#e74c3c', '#2ecc71'], startangle=90)
            plt.title('Stage 1 Wins by Objective Value: Initial vs Optimized Solution', fontsize=10)
            plt.axis('equal')
    
    # Stage 2: Quick Repair vs ALNS Replanning
    plt.subplot(1, 2, 2)
    if ('quick_repair_avg_objective' in df.columns and 
        'alns_replan_avg_objective' in df.columns):
        
        # Compare methods for each tourist
        qr_wins = 0
        alns_wins = 0
        ties = 0

        for _, row in df.iterrows():
            qr_obj = row.get('quick_repair_avg_objective')
            alns_obj = row.get('alns_replan_avg_objective')

            if not (np.isnan(qr_obj) or np.isnan(alns_obj)):
                if qr_obj > alns_obj:
                    qr_wins += 1
                elif alns_obj > qr_obj:
                    alns_wins += 1
                else:
                    ties += 1

        # Create Stage 2 pie chart
        labels = ['Quick Repair Wins', 'ALNS Replan Wins', 'Ties']
        sizes = [qr_wins, alns_wins, ties]
        colors = ['#e74c3c', '#9b59b6', '#95a5a6']

        # Filter out zero values
        filtered_labels = [label for label, size in zip(labels, sizes) if size > 0]
        filtered_sizes = [size for size in sizes if size > 0]
        filtered_colors = [color for color, size in zip(colors, sizes) if size > 0]

        if filtered_sizes:
            plt.pie(filtered_sizes, labels=filtered_labels, colors=filtered_colors, 
                    autopct='%1.1f%%', startangle=90)
            plt.title('Stage 2 Wins by Objective Value: Quick Repair vs ALNS Replan', fontsize=10)
            plt.axis('equal')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "stages_win_comparison.png"), dpi=300)
    plt.close()


def plot_stages_tourist_performance(df, output_dir="analysis_results"):
    """
    Create performance charts by tourist preferences for both Stage 1 and Stage 2.
    Uses both mean and median values.
    
    Args:
        df (pd.DataFrame): DataFrame with merged results
        output_dir (str): Directory to save output files
    """
    # Extensive debugging for Stage 2 methods
    print("\nDebugging Stage 2 Methods:")
    stage2_methods = ['quick_repair_avg_objective', 'alns_replan_avg_objective']
    for method in stage2_methods:
        print(f"\nMethod: {method}")
        if method in df.columns:
            print("Total values:", len(df[method]))
            print("Non-null values:", df[method].count())
            print("Unique values:", df[method].unique()[:10])
            print("Null values:", df[method].isnull().sum())
        else:
            print(f"Column {method} NOT FOUND in DataFrame")

    plt.figure(figsize=(15, 10))

    # Performance analysis for Stage 1 and Stage 2
    performance_stages = [
        {
            'objective_col': ['initial_objective', 'optimized_objective'],
            'method_names': ['Initial', 'Optimized'],
            'title': 'Stage 1: Performance by Preference (Initial vs Optimized)',
            'subplot': 121
        },
        {
            'objective_col': ['quick_repair_avg_objective', 'alns_replan_avg_objective'],
            'method_names': ['Quick Repair', 'ALNS Replan'],
            'title': 'Stage 2: Performance by Preference (Quick Repair vs ALNS Replan)',
            'subplot': 122
        }
    ]

    for stage_config in performance_stages:
        plt.subplot(stage_config['subplot'])
        
        # Aggregate tourists by preference
        preference_groups = defaultdict(list)

        for idx, row in df.iterrows():
            # Safely extract preferences
            try:
                prefs = row['preferences']
                
                # Handle different possible formats
                if isinstance(prefs, str):
                    # Remove quotes and brackets
                    prefs = prefs.strip("[]'")
                    # Split by comma or other separators
                    prefs = [p.strip("' ") for p in prefs.split(',')]
                elif isinstance(prefs, list):
                    # Ensure list elements are strings and stripped
                    prefs = [str(p).strip("' ") for p in prefs]
                else:
                    continue

                # Normalize preference representation
                pref_key = '/'.join(sorted(prefs))
            except Exception:
                pref_key = 'Unknown'

            # Collect objective values for the current stage
            stage_objectives = []
            for col in stage_config['objective_col']:
                try:
                    # Convert to float, handling various potential input types
                    obj_val = row.get(col)
                    
                    if pd.notna(obj_val) and obj_val != '':
                        obj_val = float(obj_val)
                        stage_objectives.append(obj_val)
                except (ValueError, TypeError):
                    # Print debugging info for problematic values
                    print(f"Problem with column {col} for preference {pref_key}")
                    print(f"Raw value: {obj_val}, Type: {type(obj_val)}")
                    continue
            
            if stage_objectives:
                preference_groups[pref_key].append(stage_objectives)

        # Calculate performance for each preference group
        pref_performance = {}
        for pref, performances in preference_groups.items():
            # Transpose performances to separate methods
            try:
                perf_array = list(map(list, zip(*performances)))
            
                # Calculate mean for each method
                mean_perfs = [np.mean(method_perfs) for method_perfs in perf_array]
                
                pref_performance[pref] = {
                    'performance': mean_perfs,
                    'count': len(performances)
                }
            except ValueError:
                # Skip this preference group if we can't transpose
                continue

        # Prepare data for plotting
        if pref_performance:
            # Sort preferences by count, take top 8
            sorted_prefs = sorted(pref_performance.items(), 
                                  key=lambda x: x[1]['count'], 
                                  reverse=True)[:8]

            # Prepare data
            pref_labels = [f"{p[0]} (n={p[1]['count']})" for p in sorted_prefs]
            method_performances = list(zip(*[p[1]['performance'] for p in sorted_prefs]))

            # Debugging performance data
            print(f"\nPerformance Data for {stage_config['title']}:")
            print("Preferences:", pref_labels)
            print("Method Performances:", method_performances)

            # Plot grouped bar chart
            x = np.arange(len(pref_labels))
            width = 0.35
            
            colors = ['#3498db', '#2ecc71']
            if len(stage_config['method_names']) == 2:
                for i, (method_name, performances) in enumerate(zip(stage_config['method_names'], method_performances)):
                    offset = width * (i - 0.5)
                    plt.bar(x + offset, performances, width, 
                            label=method_name, 
                            color=colors[i])

            plt.title(stage_config['title'], fontsize=10)
            plt.xlabel('Preference Group', fontsize=8)
            plt.ylabel('Objective Value', fontsize=8)
            plt.xticks(x, pref_labels, rotation=45, ha='right', fontsize=6)
            plt.legend(fontsize=6)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
        else:
            plt.text(0.5, 0.5, 'No performance data available', 
                     ha='center', va='center', transform=plt.gca().transAxes)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "stages_tourist_performance.png"), dpi=300)
    plt.close()

def plot_stages_objective_comparison(df, output_dir="analysis_results"):
    """
    Create bar charts comparing mean and median objective values for both stages.
    
    Args:
        df (pd.DataFrame): DataFrame with merged results
        output_dir (str): Directory to save output files
    """
    plt.figure(figsize=(15, 10))

    # Plotting methods
    methods_stage1 = ['initial_objective', 'optimized_objective']
    methods_stage2 = ['quick_repair_avg_objective', 'alns_replan_avg_objective']
    method_names_stage1 = ['Initial', 'Optimized']
    method_names_stage2 = ['Quick Repair', 'ALNS Replan']

    # Mean Objective Values (top row)
    plt.subplot(2, 1, 1)
    
    # Stage 1 Mean
    stage1_mean_values = []
    for method in methods_stage1:
        if method in df.columns:
            stage1_mean_values.append(df[method].mean())
    
    # Stage 2 Mean
    stage2_mean_values = []
    for method in methods_stage2:
        if method in df.columns:
            stage2_mean_values.append(df[method].mean())
    
    # Combine mean values
    mean_labels = method_names_stage1 + method_names_stage2
    mean_values = stage1_mean_values + stage2_mean_values
    mean_colors = ['#3498db', '#2ecc71'] + ['#e74c3c', '#9b59b6']
    
    plt.bar(mean_labels, mean_values, color=mean_colors)
    plt.title('Mean Objective Values (Stage 1 and Stage 2)', fontsize=10)
    plt.ylabel('Mean Objective Value', fontsize=8)
    plt.xticks(rotation=45, ha='right', fontsize=6)
    
    # Add values on top of bars
    for i, v in enumerate(mean_values):
        plt.text(i, v, f'{v:.4f}', ha='center', va='bottom', fontsize=8)

    # Median Objective Values (bottom row)
    plt.subplot(2, 1, 2)
    
    # Stage 1 Median
    stage1_median_values = []
    for method in methods_stage1:
        if method in df.columns:
            stage1_median_values.append(df[method].median())
    
    # Stage 2 Median
    stage2_median_values = []
    for method in methods_stage2:
        if method in df.columns:
            stage2_median_values.append(df[method].median())
    
    # Combine median values
    median_labels = method_names_stage1 + method_names_stage2
    median_values = stage1_median_values + stage2_median_values
    median_colors = ['#3498db', '#2ecc71'] + ['#e74c3c', '#9b59b6']
    
    plt.bar(median_labels, median_values, color=median_colors)
    plt.title('Median Objective Values (Stage 1 and Stage 2)', fontsize=10)
    plt.ylabel('Median Objective Value', fontsize=8)
    plt.xticks(rotation=45, ha='right', fontsize=6)
    
    # Add values on top of bars
    for i, v in enumerate(median_values):
        plt.text(i, v, f'{v:.4f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "stages_objective_comparison.png"), dpi=300)
    plt.close()

def plot_improvements_mean_median(df, output_file=None):
    """
    Calculates and plots Stage 1 and Stage 2 improvements (mean & median).

    Stage 1 improvement = (Optimized - Initial) / |Initial| * 100%
    Stage 2 improvement = (ALNS Replan - Quick Repair) / |Quick Repair| * 100%

    The function plots four bars:
        1. Stage 1 Mean Improvement
        2. Stage 1 Median Improvement
        3. Stage 2 Mean Improvement
        4. Stage 2 Median Improvement
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Make copies, dropping rows missing required columns
    df_stage1 = df.dropna(subset=['initial_objective', 'optimized_objective']).copy()
    df_stage2 = df.dropna(subset=['quick_repair_avg_objective', 'alns_replan_avg_objective']).copy()

    # Compute per-tourist improvement for Stage 1
    if not df_stage1.empty:
        df_stage1['stage1_improvement_pct'] = (
            (df_stage1['optimized_objective'] - df_stage1['initial_objective'])
            / df_stage1['initial_objective'].abs()
        ) * 100
        stage1_mean = df_stage1['stage1_improvement_pct'].mean()
        stage1_median = df_stage1['stage1_improvement_pct'].median()
    else:
        stage1_mean = np.nan
        stage1_median = np.nan

    # Compute per-tourist improvement for Stage 2
    if not df_stage2.empty:
        df_stage2['stage2_improvement_pct'] = (
            (df_stage2['alns_replan_avg_objective'] - df_stage2['quick_repair_avg_objective'])
            / df_stage2['quick_repair_avg_objective'].abs()
        ) * 100
        stage2_mean = df_stage2['stage2_improvement_pct'].mean()
        stage2_median = df_stage2['stage2_improvement_pct'].median()
    else:
        stage2_mean = np.nan
        stage2_median = np.nan

    # Prepare data for plotting
    labels = [
        'Stage 1 Mean', 
        'Stage 1 Median',
        'Stage 2 Mean',
        'Stage 2 Median'
    ]
    values = [stage1_mean, stage1_median, stage2_mean, stage2_median]

    # Set up the plot
    plt.figure(figsize=(8, 6))

    # Filter out those that are NaN (just in case)
    valid_data = [(lbl, val) for lbl, val in zip(labels, values) if not np.isnan(val)]
    if not valid_data:
        plt.text(0.5, 0.5, 'No valid improvement data found.', 
                 ha='center', va='center', transform=plt.gca().transAxes)
    else:
        valid_labels, valid_values = zip(*valid_data)
        bars = plt.bar(valid_labels, valid_values)

        # Annotate bars
        for bar, val in zip(bars, valid_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f'{val:.2f}%', ha='center', va='bottom')

        plt.title('Stage 1 & Stage 2 Improvements (Mean & Median)', fontsize=14)
        plt.ylabel('Improvement (%)', fontsize=12)
        plt.xticks(rotation=30, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')

    plt.close()
            
def main():
    """Main function to run the comprehensive analysis."""
    print("Starting comprehensive analysis of tourism itinerary planning results...")

    # Create output directory
    os.makedirs("analysis_results", exist_ok=True)

    # Analyze Stage 1 results
    print("Analyzing Stage 1 results...")
    stage1_df = analyze_stage1_results()

    if not stage1_df.empty:
        print(f"Found {len(stage1_df)} Stage 1 results.")
    else:
        print("No valid Stage 1 results found.")

    # Analyze Stage 2 results
    print("Analyzing Stage 2 results...")
    stage2_df = analyze_stage2_results()

    if not stage2_df.empty:
        print(f"Found {len(stage2_df)} Stage 2 results.")
    else:
        print("No valid Stage 2 results found.")

    # Merge results
    print("Merging Stage 1 and Stage 2 results...")
    combined_df = merge_stage_results(stage1_df, stage2_df)

    if combined_df.empty:
        print("WARNING: No valid data found in either Stage 1 or Stage 2 results.")
        return

    # # Save the combined DataFrame to CSV for further analysis
    # combined_df.to_csv("analysis_results/combined_metrics.csv")
    # print(f"Combined metrics saved to analysis_results/combined_metrics.csv")

    # Generate visualizations
    print("Generating visualizations...")
    plot_stages_win_comparison(combined_df)
    plot_stages_objective_comparison(combined_df)
    plot_stages_tourist_performance(combined_df)
    plot_improvements_mean_median(combined_df, "analysis_results/stages_improvements_mean_median.png"
)

    # Generate summary statistics
    print("Generating summary statistics...")
    summary_text = generate_summary_statistics(combined_df, "analysis_results/summary_statistics.txt")
    
    # Export cases where ALNS didn't improve
    export_non_improving_cases(combined_df, "analysis_results")

    print("\nAnalysis complete. Results saved to 'analysis_results' directory.")
    print("\nSummary of findings:")
    print("-" * 40)

    # Print an excerpt of the summary
    summary_lines = summary_text.split('\n')
    for i, line in enumerate(summary_lines):
        if i < 15 or i > len(summary_lines) - 10:
            print(line)
        elif i == 15:
            print("...")

    

if __name__ == "__main__":
    main()