#%%

# research_pipeline.py
import kagglehub
import requests
from dataclasses import dataclass
from datasets import load_dataset, DatasetDict, Dataset
from typing import List, Dict, Any, Optional
import json
import random
import openai
import os
from pathlib import Path
import pandas as pd

"""
This script forms the high-level data pipeline for investigating how well an LLM can articulate in natural language rules that it uses for a classification task.

The pipeline includes:
- Dataset downloading or creation
- Data preparation
- Running classification experiments
- Testing articulation of classification rules
- Evaluating results
"""

from datasets import load_dataset, DatasetDict, Dataset
from random import sample
from typing import List
from rich import print

# model = "gpt-4o-mini"
model = "gpt-4o"

def create_lowercase_vs_capitalized_dataset() -> Dataset:
    """
    Creates a dataset with 1,000 lowercase strings and 1,000 capitalized strings,
    labeled as 'false' or 'true' based on their casing.

    Returns:
        DatasetDict: A dictionary containing the training and testing datasets.
    """
    # Load 2,000 unique data points from the 'ag_news' dataset
    sample_dataset = load_dataset("ag_news", split='train[:2000]')

    first_entry = sample_dataset[0]
    print(first_entry)

    lowercase_data = [{'text': entry.lower(), 'label': True} for entry in sample_dataset['text'][:1000]]
    capitalized_data = [{'text': entry.capitalize(), 'label': False} for entry in sample_dataset['text'][1000:2000]]
    
    all_data = lowercase_data + capitalized_data
    
    # Create a dataset
    dataset = Dataset.from_list(all_data)
    
    # Shuffle the dataset to mix 'true' and 'false' labels
    dataset = dataset.shuffle(seed=42)
    
    return dataset

def create_lowercase_vs_allcaps_dataset() -> Dataset:
    """
    Creates a dataset with 1,000 lowercase strings and 1,000 ALL CAPS strings,
    labeled as 'false' or 'true' based on their casing.

    Returns:
        DatasetDict: A dictionary containing the training and testing datasets.
    """
    # Load 2,000 unique data points from the 'ag_news' dataset
    sample_dataset = load_dataset("ag_news", split='train[:2000]')

    first_entry = sample_dataset[0]
    print(first_entry)

    lowercase_data = [{'text': entry.lower(), 'label': True} for entry in sample_dataset['text'][:1000]]
    allcaps_data = [{'text': entry.upper(), 'label': False} for entry in sample_dataset['text'][1000:2000]]
    
    all_data = lowercase_data + allcaps_data
    
    # Create a dataset
    dataset = Dataset.from_list(all_data)
    
    # Shuffle the dataset to mix 'true' and 'false' labels
    dataset = dataset.shuffle(seed=42)
    
    return dataset

def create_few_shot_prompts(dataset: Dataset) -> List[str]:
    """
    Creates few-shot classification prompts from the dataset until it's exhausted.

    Args:
        dataset (Dataset): The dataset containing data and labels.

    Returns:
        List[str]: A list of formatted few-shot classification prompts.
    """
    prompts = []
    num_examples = 20  # Number of few-shot examples to include

    # Iterate through the dataset in chunks of num_examples
    for start_idx in range(0, len(dataset), num_examples + 1):
        # Ensure we have enough data for few-shot examples and a test example
        if start_idx + num_examples >= len(dataset):
            break
        
        # Split the dataset into few-shot examples and test example
        few_shot_examples = dataset.select(range(start_idx, start_idx + num_examples))
        test_example = dataset[start_idx + num_examples]
        
        # Build the prompt
        prompt = ""
        for idx, example in enumerate(few_shot_examples, start=1):
            prompt += f"Example {idx}:\n"
            prompt += f"data{idx}: {example['text']}\n"
            prompt += f"label{idx}: {example['label']}\n\n"
        
        next_idx = num_examples + 1
        final_prompt_part = f"data{next_idx}: {test_example['text']}\n\n"
        final_prompt_part += """
There is a clear and simple rule that is used to classify the data.
data21 similarly can be classified using the same rule.
Think step-by-step what this rule is, then respond in this format:
<response>true</response> or <response>false</response>"""

        label = test_example['label']
        
        prompts.append((prompt, final_prompt_part, label))

    return prompts

def normalize_response(response: str) -> bool:
    """Extract boolean from XML response."""
    import re
    match = re.search(r'<response>(true|false)</response>', response.lower())
    if match:
        return match.group(1) == 'true'
    raise ValueError(f"Could not parse XML response: {response}")

#%% 

def send_prompt(prompt:str) -> str:
    res = openai.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    if res.choices[0].message.content is None:
        return ""
    return res.choices[0].message.content


# %%

# i want to create a dataset of caesar cipher vs random substitution cipher
# start from a basic text dataset, strip out everything that's not a letter
# then apply a caesar cipher and a random substitution cipher to generate two entries in the dataset,
# label the first one true and the second one false

def create_caesar_vs_random_substitution_dataset() -> Dataset:
    # Start with some basic English text samples
    texts = []
    # Load some sample texts
    dataset = load_dataset('ag_news')
    texts = [text for text in dataset['train']['text']][:1000]  # Take first 100 texts
    dataset = []
    
    for text in texts:
        # Keep spaces and convert to lowercase
        cleaned_text = ''.join(c.lower() for c in text if c.isalpha() or c.isspace())
        
        # Create Caesar cipher version (shift by random amount)
        shift = random.randint(1, 25)
        caesar_text = ''
        for c in cleaned_text:
            if c.isalpha():  # Shift only letters
                # Shift each character by the random amount
                shifted = chr(((ord(c) - ord('a') + shift) % 26) + ord('a'))
                caesar_text += shifted
            else:
                caesar_text += c  # Keep spaces as they are
            
        # Create random substitution cipher version
        alphabet = list('abcdefghijklmnopqrstuvwxyz')
        substitution_map = dict(zip(alphabet, random.sample(alphabet, 26)))
        substitution_text = ''.join(substitution_map[c] if c.isalpha() else c for c in cleaned_text)
        
        # Add both to dataset with appropriate labels
        dataset.append((caesar_text, True))  # Caesar cipher labeled as True
        dataset.append((substitution_text, False))  # Random substitution labeled as False
        
    # Convert the list of tuples into a list of dictionaries
    formatted_data = [
        {'text': text, 'label': label} 
        for text, label in dataset
    ]
    
    # Create and return a Dataset object instead of a list
    return Dataset.from_list(formatted_data)


def create_caesar_vs_random_letters_dataset() -> Dataset:
    # Start with some basic English text samples
    texts = []
    # Load some sample texts
    dataset = load_dataset('ag_news')
    texts = [text for text in dataset['train']['text']][:1000]  # Take first 100 texts
    dataset = []
    
    for text in texts:
        # Keep spaces and convert to lowercase
        cleaned_text = ''.join(c.lower() for c in text if c.isalpha() or c.isspace())
        
        # Create Caesar cipher version (shift by random amount)
        shift = random.randint(1, 25)
        caesar_text = ''
        for c in cleaned_text:
            if c.isalpha():  # Shift only letters
                # Shift each character by the random amount
                shifted = chr(((ord(c) - ord('a') + shift) % 26) + ord('a'))
                caesar_text += shifted
            else:
                caesar_text += c  # Keep spaces as they are
            
        # Create random letters version (completely random letters)
        random_text = ''
        for c in cleaned_text:
            if c.isalpha():
                # Replace each letter with a random letter
                random_text += random.choice('abcdefghijklmnopqrstuvwxyz')
            else:
                random_text += c  # Keep spaces as they are
        
        # Add both to dataset with appropriate labels
        dataset.append((caesar_text, True))  # Caesar cipher labeled as True
        dataset.append((random_text, False))  # Random letters labeled as False
        
    # Convert the list of tuples into a list of dictionaries
    formatted_data = [
        {'text': text, 'label': label} 
        for text, label in dataset
    ]
    
    # Create and return a Dataset object instead of a list
    return Dataset.from_list(formatted_data)

def create_happy_vs_sad_dataset() -> Dataset:
    """
    Creates a dataset with happy and sad text samples from the emotion dataset,
    labeled as True for happy and False for sad.

    Returns:
        Dataset: A dataset containing text samples labeled as happy (True) or sad (False)
    """
    # Load the emotion dataset
    dataset = load_dataset('emotion', split='train')
    
    # In the emotion dataset:
    # Label 3 = Joy/Happy
    # Label 4 = Sadness
    happy_samples = [(text, True) for text, label in zip(dataset['text'], dataset['label']) 
                    if label == 3][:1000]  # Get first 1000 happy samples
    sad_samples = [(text, False) for text, label in zip(dataset['text'], dataset['label']) 
                   if label == 4][:1000]  # Get first 1000 sad samples
    
    # Combine happy and sad samples
    all_samples = happy_samples + sad_samples
    
    # Convert to format expected by Dataset.from_list
    formatted_data = [
        {'text': text, 'label': label}
        for text, label in all_samples
    ]
    
    # Create dataset and shuffle
    dataset = Dataset.from_list(formatted_data)
    dataset = dataset.shuffle(seed=42)
    
    return dataset

def create_emotion_tweets_dataset() -> Dataset:
    """
    Creates a dataset with happy vs sad tweets,
    labeled as True for happy (label 3) and False for sad (label 4).
    
    Label meanings in the dataset:
    0 = neutral
    1 = worry
    2 = happiness
    3 = sadness
    4 = love
    5 = fun
    """
    # Download latest version
    path = kagglehub.dataset_download("aadyasingh55/twitter-emotion-classification-dataset")
    
    # Load the downloaded dataset - find the parquet file
    parquet_path = next(Path(path).glob("*.parquet"))
    df = pd.read_parquet(parquet_path)
    
    # Debug prints
    print(f"Loaded dataframe with shape: {df.shape}")
    print(f"Columns: {df.columns}")
    print(f"First few rows:\n{df.head()}")
    
    # Print unique labels and their counts
    print("\nLabel distribution:")
    print(df['label'].value_counts())
    
    # Convert labels to integers if they're strings
    df['label'] = df['label'].astype(int)
    
    # Select happiness (2) vs sadness (3) tweets
    happy_samples = df[df['label'] == 2][:1000]  # Get first 1000 happy samples
    sad_samples = df[df['label'] == 3][:1000]    # Get first 1000 sad samples
    
    print(f"\nSelected {len(happy_samples)} happy samples and {len(sad_samples)} sad samples")
    
    # Combine and format the data
    formatted_data = (
        [{'text': text, 'label': True} for text in happy_samples['text']] +
        [{'text': text, 'label': False} for text in sad_samples['text']]
    )
    
    # Create dataset and shuffle
    dataset = Dataset.from_list(formatted_data)
    dataset = dataset.shuffle(seed=42)
    
    return dataset


def create_ai_vs_human_dataset() -> Dataset:
    # Download latest version
    path = kagglehub.dataset_download("sunilthite/llm-detect-ai-generated-text-dataset")
    
    # Load the downloaded dataset - find the parquet file
    print(f"Path to dataset files: {path}")
    csv_path = next(Path(path).glob("*.csv"))
    df = pd.read_csv(csv_path)
    # Debug prints
    print(f"Loaded dataframe with shape: {df.shape}")
    print(f"Columns: {df.columns}")
    print(f"First few rows:\n{df.head()}")
    
    # Print unique labels and their counts
    print("\nLabel distribution:")
    print(df['generated'].value_counts())
    
    # Select AI generated (True) vs human (False) texts
    ai_samples = df[df['generated'] == 1][:1000]  # Get first 1000 AI samples
    human_samples = df[df['generated'] == 0][:1000]  # Get first 1000 human samples
    
    print(f"\nSelected {len(ai_samples)} AI samples and {len(human_samples)} human samples")
    
    # Combine and format the data
    formatted_data = (
        [{'text': text, 'label': True} for text in ai_samples['text']] +
        [{'text': text, 'label': False} for text in human_samples['text']]
    )
    
    # Create dataset and shuffle
    dataset = Dataset.from_list(formatted_data)
    dataset = dataset.shuffle(seed=42)
    
    return dataset


def create_hendrycks_commonsense_dataset() -> Dataset:
    """
    Creates a dataset from the Hendrycks ethics commonsense dataset,
    with a more efficient data processing approach.
    """
    ds = load_dataset("hendrycks/ethics", "commonsense")
    train_data = ds['train']
    
    # Convert to list format all at once instead of iterating
    formatted_data = [
        {
            'text': input_text,
            'label': bool(label)
        }
        for input_text, label in zip(train_data['input'], train_data['label'])
    ]
    
    # Create dataset and shuffle
    dataset = Dataset.from_list(formatted_data)
    dataset = dataset.shuffle(seed=42)
    
    # Optionally limit the size to match other datasets
    if len(dataset) > 2000:
        dataset = dataset.select(range(2000))
    
    return dataset


def create_language_pair_dataset(lang_pair: str) -> Dataset:
    """
    Creates a dataset with text samples in two languages,
    labeled as True for the first language and False for the second.

    Args:
        lang_pair: Language pair code (e.g. 'fr-en' or 'es-en') from OPUS Books dataset

    Returns:
        Dataset: A dataset containing text samples labeled as first language (True) 
                or second language (False)
    """
    # Load OPUS Books dataset with the specified language pair
    dataset = load_dataset("opus_books", lang_pair)
    
    # Get the language codes
    lang1, lang2 = lang_pair.split('-')
    
    # Initialize lists for samples
    lang1_samples = []
    lang2_samples = []
    
    # Process the training split
    for item in dataset['train']:
        # Get the translations as single strings
        text1 = item['translation'][lang1]
        text2 = item['translation'][lang2]
        
        if len(lang1_samples) < 1000:
            lang1_samples.append((text1, True))
        if len(lang2_samples) < 1000:
            lang2_samples.append((text2, False))
        
        if len(lang1_samples) >= 1000 and len(lang2_samples) >= 1000:
            break
    
    # Combine samples
    all_samples = lang1_samples + lang2_samples
    
    # Convert to format expected by Dataset.from_list
    formatted_data = [
        {'text': text, 'label': label}
        for text, label in all_samples
    ]
    
    # Create dataset and shuffle
    dataset = Dataset.from_list(formatted_data)
    dataset = dataset.shuffle(seed=42)
    
    return dataset

# %%

def create_python_vs_javascript_dataset() -> Dataset:
    """
    Creates a dataset with Python and JavaScript code samples,
    labeled as True for Python and False for JavaScript.

    Returns:
        Dataset: A dataset containing code samples labeled as Python (True) 
                or JavaScript (False)
    """
    # Load CodeSearchNet dataset for both languages
    python_ds = load_dataset("code_search_net", "python", split="train")
    javascript_ds = load_dataset("code_search_net", "javascript", split="train")
    
    # Get code samples (using 'whole_func_string' which contains the complete function)
    python_samples = [(code, True) for code in python_ds["whole_func_string"][:1000]]
    javascript_samples = [(code, False) for code in javascript_ds["whole_func_string"][:1000]]
    
    # Combine samples
    all_samples = python_samples + javascript_samples
    
    # Convert to format expected by Dataset.from_list
    formatted_data = [
        {'text': text, 'label': label}
        for text, label in all_samples
    ]
    
    # Create dataset and shuffle
    dataset = Dataset.from_list(formatted_data)
    dataset = dataset.shuffle(seed=42)
    
    return dataset


# %%

def load_or_create_results(dataset_name: str, prompts: List[tuple]) -> List[dict]:
    """Load cached results or create new ones if they don't exist."""
    cache_dir = Path('.cache/results')
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{dataset_name}_{model}_results.json"
    
    # Try to load cached results
    if cache_file.exists():
        with open(cache_file, 'r') as f:
            return json.load(f)
    
    # If no cache, create new results
    responses = []
    correct = 0
    total = 0

    # just for testing
    # prompts = prompts[:20]
    
    from concurrent.futures import ThreadPoolExecutor
    from tqdm import tqdm

    def process_prompt(args):
        idx, (prompt, final_prompt_part, label) = args
        classification_prompt = prompt + final_prompt_part
        res = send_prompt(classification_prompt)
        articulation_prompt = prompt + """
        There is a clear and simple rule that is used to classify the data. 
        Think it through step by step and then respond with your final answer
        for what the rule is.
        """
        articulation_res = send_prompt(articulation_prompt)

        try:
            normalized = normalize_response(res)
            return {
                'prompt': classification_prompt,
                'response': res, 
                'true_label': label,
                'normalized_response': normalized,
                'is_correct': normalized == label,
                'rule_articulation': articulation_res,
                'success': True
            }
        except ValueError as e:
            print(f"Warning: Could not parse response: {e}")
            return {'success': False}

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = list(tqdm(
            executor.map(process_prompt, enumerate(prompts)), 
            total=len(prompts),
            desc="Processing prompts"
        ))

    for result in futures:
        if result['success']:
            responses.append(result)
            total += 1
            if result['is_correct']:
                correct += 1
    
    # Save results to cache
    with open(cache_file, 'w') as f:
        json.dump(responses, f)
    
    print(f"Accuracy: {correct}/{total} ({(correct/total)*100:.2f}%)")
    return responses
    

# Replace your dataset loop with:
datasets = [
    ('caesar_vs_substitution', create_caesar_vs_random_substitution_dataset()),
    ('lowercase_vs_capitalized', create_lowercase_vs_capitalized_dataset()),
    ('caesar_vs_random', create_caesar_vs_random_letters_dataset()),
    ('happy_vs_sad', create_happy_vs_sad_dataset()),
    ('emotion_tweets', create_emotion_tweets_dataset()),
    ('ai_vs_human', create_ai_vs_human_dataset()),
    ('hendrycks_commonsense', create_hendrycks_commonsense_dataset()),
    ('french_vs_english', create_language_pair_dataset('en-fr')),
    ('spanish_vs_english', create_language_pair_dataset('en-es')),
    ('python_vs_javascript', create_python_vs_javascript_dataset()),
    ('lowercase_vs_allcaps', create_lowercase_vs_allcaps_dataset()),
]

for dataset in datasets:
    # print first 3 entries in dataset
    print(f"first 3 entries in {dataset[0]}:")
    print(dataset[1][:3])

for dataset_name, dataset in datasets:
    few_shot_prompts = create_few_shot_prompts(dataset)
    print(f'{len(few_shot_prompts)} few-shot prompts created for {dataset.info.description}.')
    
    responses = load_or_create_results(dataset_name, few_shot_prompts)
    
    # Calculate and print accuracy from cached results
    correct = sum(1 for r in responses if r['is_correct'])
    total = len(responses)
    print(f"Accuracy: {correct}/{total} ({(correct/total)*100:.2f}%)")



# %%

# for response in responses[:3]:
#     print(response['rule_articulation'])

# %%
# # Load and display first 3 prompts from Spanish vs English results
# with open('.cache/results/spanish_vs_english_results.json', 'r') as f:
#     spanish_english_responses = json.load(f)
    
# print("First 3 prompts from Spanish vs English dataset:")
# for response in spanish_english_responses[:3]:
#     print("\nPrompt:", response['prompt'], "\nExplanation:", response.get('rule_articulation', 'No explanation provided.'))

with open('.cache/results/lowercase_vs_allcaps_results.json', 'r') as f:
    lowercase_allcaps_responses = json.load(f)

for response in lowercase_allcaps_responses[:3]:
    print("\nPrompt:", response['prompt'], "\nExplanation:", response.get('rule_articulation', 'No explanation provided.'))


# %%
