import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_accuracy_by_articulation(data_dir: str | Path):
    """
    Create two horizontal bar charts showing accuracy for each task at 100% articulation,
    split by model (gpt-4o and gpt-4o-mini).
    
    Args:
        data_dir: Directory containing the JSON result files
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    # Initialize data list
    data = []
    
    # Get all JSON files in directory
    data_dir = Path(data_dir)
    json_files = list(data_dir.glob('*.json'))
    
    # Extract data from each file
    for file_path in json_files:
        with open(file_path) as f:
            json_data = json.load(f)
            filename = file_path.stem  # Get filename without extension
            
            for entry in json_data:
                for response in entry['responses']:
                    # Only include responses with both percentage and correctness
                    if 'articulation_percentage' in response and 'classified_correctly' in response:
                        data.append({
                            'filename': filename,
                            'articulation_percentage': response['articulation_percentage'],
                            'classified_correctly': response['classified_correctly']
                        })
    
    # Create DataFrame and filter for full articulation
    df = pd.DataFrame(data)
    df = df[df['articulation_percentage'] == 1.0]
    
    # Split into two dataframes based on model
    mini_df = df[df['filename'].str.contains('mini')].copy()
    gpt4_df = df[~df['filename'].str.contains('mini')].copy()
    
    # Calculate accuracy for each and sort by accuracy (descending)
    mini_accuracy = mini_df.groupby('filename')['classified_correctly'].mean().reset_index()
    gpt4_accuracy = gpt4_df.groupby('filename')['classified_correctly'].mean().reset_index()
    
    # Sort both dataframes by accuracy (descending)
    mini_accuracy = mini_accuracy.sort_values('classified_correctly', ascending=True)  # True for horizontal bars
    gpt4_accuracy = gpt4_accuracy.sort_values('classified_correctly', ascending=True)
    
    # Create the plot with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
    
    # Plot GPT-4
    ax1.barh(gpt4_accuracy['filename'], gpt4_accuracy['classified_correctly'])
    ax1.axvline(x=0.9, color='black', linestyle=':', alpha=0.5)  # Add dotted line
    ax1.set_xlabel('Accuracy')
    ax1.set_title('GPT-4 Accuracy (100% articulation)')
    
    # Plot GPT-4-mini
    ax2.barh(mini_accuracy['filename'], mini_accuracy['classified_correctly'])
    ax2.axvline(x=0.9, color='black', linestyle=':', alpha=0.5)  # Add dotted line
    ax2.set_xlabel('Accuracy')
    ax2.set_title('GPT-4-mini Accuracy (100% articulation)')
    
    plt.tight_layout()
    return fig

def plot_accuracy_by_all_articulations(data_dir: str | Path):
    """
    Create two line plots showing accuracy for each task across all articulation percentages,
    split by model (gpt-4 and gpt-4-mini).
    
    Args:
        data_dir: Directory containing the JSON result files
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    # Initialize data list
    data = []
    
    # Get all JSON files in directory
    data_dir = Path(data_dir)
    json_files = list(data_dir.glob('*.json'))
    
    # Extract data from each file
    for file_path in json_files:
        with open(file_path) as f:
            json_data = json.load(f)
            filename = file_path.stem
            
            for entry in json_data:
                for response in entry['responses']:
                    if 'articulation_percentage' in response and 'classified_correctly' in response:
                        data.append({
                            'filename': filename,
                            'articulation_percentage': response['articulation_percentage'],
                            'classified_correctly': response['classified_correctly']
                        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Split into two dataframes based on model
    mini_df = df[df['filename'].str.contains('mini')].copy()
    gpt4_df = df[~df['filename'].str.contains('mini')].copy()
    
    # Calculate accuracy for each combination of filename and articulation percentage
    mini_accuracy = mini_df.groupby(['filename', 'articulation_percentage'])['classified_correctly'].mean().reset_index()
    gpt4_accuracy = gpt4_df.groupby(['filename', 'articulation_percentage'])['classified_correctly'].mean().reset_index()
    
    # Create the plot with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # Plot GPT-4
    for filename in gpt4_accuracy['filename'].unique():
        task_data = gpt4_accuracy[gpt4_accuracy['filename'] == filename]
        ax1.plot(task_data['articulation_percentage'], 
                task_data['classified_correctly'], 
                marker='o', 
                label=filename)
    
    ax1.axhline(y=0.9, color='black', linestyle=':', alpha=0.5)  # Add dotted line
    ax1.set_xlabel('Articulation Percentage')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('GPT-4o Accuracy by Articulation Percentage')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot GPT-4-mini
    for filename in mini_accuracy['filename'].unique():
        task_data = mini_accuracy[mini_accuracy['filename'] == filename]
        ax2.plot(task_data['articulation_percentage'], 
                task_data['classified_correctly'], 
                marker='o', 
                label=filename)
    
    ax2.axhline(y=0.9, color='black', linestyle=':', alpha=0.5)  # Add dotted line
    ax2.set_xlabel('Articulation Percentage')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('GPT-4-mini Accuracy by Articulation Percentage')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_correct_counts_by_articulation(data_dir: str | Path):
    """
    Create two line plots showing the number of correct classifications for each task 
    across all articulation percentages, split by model (gpt-4 and gpt-4-mini).
    
    Args:
        data_dir: Directory containing the JSON result files
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    # Initialize data list
    data = []
    
    # Get all JSON files in directory
    data_dir = Path(data_dir)
    json_files = list(data_dir.glob('*.json'))
    
    # Extract data from each file
    for file_path in json_files:
        with open(file_path) as f:
            json_data = json.load(f)
            filename = file_path.stem
            
            for entry in json_data:
                for response in entry['responses']:
                    if 'articulation_percentage' in response and 'classified_correctly' in response:
                        data.append({
                            'filename': filename,
                            'articulation_percentage': response['articulation_percentage'],
                            'classified_correctly': response['classified_correctly']
                        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Split into two dataframes based on model
    mini_df = df[df['filename'].str.contains('mini')].copy()
    gpt4_df = df[~df['filename'].str.contains('mini')].copy()
    
    # Calculate correct counts for each combination of filename and articulation percentage
    mini_correct_counts = mini_df.groupby(['filename', 'articulation_percentage'])['classified_correctly'].sum().reset_index()
    gpt4_correct_counts = gpt4_df.groupby(['filename', 'articulation_percentage'])['classified_correctly'].sum().reset_index()
    
    # Create the plot with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # Plot GPT-4
    for filename in gpt4_correct_counts['filename'].unique():
        task_data = gpt4_correct_counts[gpt4_correct_counts['filename'] == filename]
        ax1.plot(task_data['articulation_percentage'], 
                task_data['classified_correctly'], 
                marker='o', 
                label=filename)
    
    ax1.axhline(y=0.9, color='black', linestyle=':', alpha=0.5)  # Add dotted line
    ax1.set_xlabel('Articulation Percentage')
    ax1.set_ylabel('Correct Counts')
    ax1.set_title('GPT-4o Correct Counts by Articulation Percentage')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot GPT-4-mini
    for filename in mini_correct_counts['filename'].unique():
        task_data = mini_correct_counts[mini_correct_counts['filename'] == filename]
        ax2.plot(task_data['articulation_percentage'], 
                task_data['classified_correctly'], 
                marker='o', 
                label=filename)
    
    ax2.axhline(y=0.9, color='black', linestyle=':', alpha=0.5)  # Add dotted line
    ax2.set_xlabel('Articulation Percentage')
    ax2.set_ylabel('Correct Counts')
    ax2.set_title('GPT-4-mini Correct Counts by Articulation Percentage')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_correct_counts_individual(data_dir: str | Path):
    """
    Create multiple subplots (arranged in rows of 3) showing the number of correct 
    classifications for each task across all articulation percentages. Each task gets
    its own subplot, with both GPT-4 and GPT-4-mini versions on the same plot.
    
    Args:
        data_dir: Directory containing the JSON result files
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    # Initialize data list
    data = []
    
    # Get all JSON files in directory
    data_dir = Path(data_dir)
    json_files = list(data_dir.glob('*.json'))
    
    # Extract data from each file
    for file_path in json_files:
        with open(file_path) as f:
            json_data = json.load(f)
            filename = file_path.stem
            
            for entry in json_data:
                for response in entry['responses']:
                    if 'articulation_percentage' in response and 'classified_correctly' in response:
                        data.append({
                            'filename': filename,
                            'articulation_percentage': response['articulation_percentage'],
                            'classified_correctly': response['classified_correctly']
                        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Split into two dataframes based on model
    mini_df = df[df['filename'].str.contains('mini')].copy()
    gpt4_df = df[~df['filename'].str.contains('mini')].copy()
    
    # Count correct classifications for each combination
    mini_counts = mini_df[mini_df['classified_correctly']].groupby(
        ['filename', 'articulation_percentage']).size().reset_index(name='correct_count')
    gpt4_counts = gpt4_df[gpt4_df['classified_correctly']].groupby(
        ['filename', 'articulation_percentage']).size().reset_index(name='correct_count')
    
    # Get unique task names (without mini suffix)
    tasks = sorted(set(name.replace('-mini', '') for name in df['filename'].unique()))
    num_tasks = len(tasks)
    
    # Calculate number of rows needed (3 plots per row)
    num_rows = (num_tasks + 2) // 3
    
    # Create figure with subplots - reduced figure size
    fig, axes = plt.subplots(num_rows, 3, figsize=(12, 3*num_rows))
    axes = axes.flatten()  # Flatten to make indexing easier
    
    # Plot each task
    for idx, task in enumerate(tasks):
        ax = axes[idx]
        
        # Plot GPT-4 data
        gpt4_task_data = gpt4_counts[gpt4_counts['filename'] == task].sort_values('articulation_percentage')
        if not gpt4_task_data.empty:
            ax.plot(gpt4_task_data['articulation_percentage'], 
                   gpt4_task_data['correct_count'],
                   'o-',  # Line with circle markers
                   linewidth=1.5,
                   markersize=4,
                   label='GPT-4')
            
            # Find the closest value to 0.1 articulation percentage
            closest_to_01 = gpt4_task_data.iloc[(gpt4_task_data['articulation_percentage'] - 0.1).abs().argsort()[:1]]
            val_at_01 = closest_to_01['correct_count'].iloc[0]
            
            # Set y-axis limits to ±20% of that value
            y_min = val_at_01 * 0.8
            y_max = val_at_01 * 1.2
            ax.set_ylim(y_min, y_max)
        
        # Plot GPT-4-mini data
        mini_task = f"{task}-mini"
        mini_task_data = mini_counts[mini_counts['filename'] == mini_task].sort_values('articulation_percentage')
        if not mini_task_data.empty:
            ax.plot(mini_task_data['articulation_percentage'], 
                   mini_task_data['correct_count'],
                   's-',  # Line with square markers
                   linewidth=1.5,
                   markersize=4,
                   label='GPT-4-mini')
        
        ax.set_title(task, fontsize=10)
        ax.set_xlabel('Articulation Percentage', fontsize=8)
        ax.set_ylabel('Correct Count', fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        
        # Set x-axis limits to 0-1
        ax.set_xlim(-0.1, 1.1)
    
    # Remove any empty subplots
    for idx in range(num_tasks, len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    return fig


fig = plot_correct_counts_individual('.cache/results')
plt.show()

