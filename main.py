#%%

# research_pipeline.py
import kagglehub
import requests
from dataclasses import dataclass
from datasets import load_dataset, DatasetDict, Dataset
from typing import List, Dict, Any, Optional, Union
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

models = ["gpt-4o", "gpt-4o-mini"]
model = models[1]

@dataclass
class AnnotatedDataset:
    name: str
    dataset: Any  # Could be more specific based on your dataset type
    true_label: int
    articulations: list[str]

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

@dataclass
class PromptExample:
    prompt: str
    data21: str
    label: bool

def create_few_shot_prompts(annotated_dataset: AnnotatedDataset) -> List[PromptExample]:
    """
    Creates few-shot classification prompts from the annotated dataset until it's exhausted.

    Args:
        annotated_dataset (AnnotatedDataset): The annotated dataset containing data, labels, and articulations.

    Returns:
        List[PromptExample]: A list of formatted few-shot classification prompts.
    """
    dataset = annotated_dataset.dataset
    prompts = []
    num_examples = 20  # Number of few-shot examples to include

    for start_idx in range(0, len(dataset), num_examples + 1):
        if start_idx + num_examples >= len(dataset):
            break
        
        few_shot_examples = dataset.select(range(start_idx, start_idx + num_examples))
        test_example = dataset[start_idx + num_examples]
        
        prompt = ""
        prompt += "\nNow, here are some examples:\n\n"
        
        for idx, example in enumerate(few_shot_examples, start=1):
            prompt += f"Example {idx}:\n"
            prompt += f"data{idx}: {example['text']}\n"
            prompt += f"label{idx}: {example['label']}\n\n"
        
        data21 = f"data21: {test_example['text']}"
        label = test_example['label']
        
        prompts.append(PromptExample(prompt=prompt, data21=data21, label=label))

    print(f"Created {len(prompts)} prompts")
    print(prompts[0])
    return prompts

def normalize_response(response: str) -> Union[bool, int]:
    """Extract boolean or integer from XML response."""
    import re
    # Try to match boolean response first
    bool_match = re.search(r'<response>(true|false)</response>', response.lower())
    if bool_match:
        return bool_match.group(1) == 'true'
    
    # Try to match numeric response and convert to 0-based
    num_match = re.search(r'<response>(\d+)</response>', response.lower())
    if num_match:
        # Convert 1-based input to 0-based
        return int(num_match.group(1))
        
    print(f"Could not parse XML response: {response}")
    return -1

#%% 

def send_messages(messages: List[dict]) -> str:
    res = openai.chat.completions.create(
        model=model,
        messages=messages,
    )
    return res.choices[0].message.content

def send_prompt_to_res(prompt:str):
    res = openai.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return res

def send_prompt_to_text(prompt:str) -> str:
    res = send_prompt_to_res(prompt)
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

def load_or_create_results(dataset: AnnotatedDataset, prompts: List[PromptExample]) -> List[dict]:
    """Load cached results or create new ones if they don't exist."""
    cache_dir = Path('.cache/results')
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{dataset.name}_{model}_cot_results.json"
    
    # Try to load cached results
    if cache_file.exists():
        with open(cache_file, 'r') as f:
            return json.load(f)
    
    # If no cache, create new results
    responses = []

    # just for testing
    # prompts = prompts[:20]
    
    from concurrent.futures import ThreadPoolExecutor
    from tqdm import tqdm

    def process_prompt(args):
        idx, prompt_example = args
        articulation_prompt = prompt_example.prompt + """
There is a clear and simple rule that is used to classify the data. Think step-by-step what this rule is."
        """
        # Get the full articulation first
        articulation = send_prompt_to_text(articulation_prompt)

        # Create MCQ prompt with the actual articulations
        articulation_mcq_prompt = prompt_example.prompt + f"""
You are given a list of possible articulations of a classification rule:

{chr(10).join(f"{i}. {art}" for i, art in enumerate(dataset.articulations))}

Based on the examples you've seen, which of these articulations (0-{len(dataset.articulations)-1}) best describes the true classification rule? 
Respond with just the number in the following format: <response>{random.randint(0,3)}</response> or <response>{random.randint(0,3)}</response> etc.
"""
        articulation_mcq = send_prompt_to_text(articulation_mcq_prompt)
        normalized_mcq = normalize_response(articulation_mcq)
        
        # Calculate if the articulation was correct
        articulation_correct = normalized_mcq == dataset.true_label

        percentage_to_cut_short = [0, 0.33, 0.66, 0.9]
        responses_per_articulation = []
        
        # Generate shortened versions and get responses for each
        for pct in percentage_to_cut_short:
            if pct == 0:
                current_articulation = articulation
            else:
                chars_to_keep = int(len(articulation) * (1 - pct))
                current_articulation = articulation[:chars_to_keep] + "..."
                
            # Get classification response for this articulation
            msg2 = f"""
Thank you, now please classify the following data21. {prompt_example.data21}
Give your answer in this format: <response>true</response> or <response>false</response>.
The first word of your response must be <response>, no yapping.
"""
            messages = [
                {"role": "user", "content": articulation_prompt},
                {"role": "assistant", "content": current_articulation},
                {"role": "user", "content": msg2}
            ]
            res = send_messages(messages)
            
            try:
                normalized = normalize_response(res)
                responses_per_articulation.append({
                    'response_text': res,
                    'normalized_response': normalized,
                    'classified_correctly': normalized == prompt_example.label,
                    'articulation_length': len(current_articulation),
                    'articulation_percentage': 1 - pct,
                    'articulation': current_articulation
                })
            except ValueError as e:
                print(f"Warning: Could not parse response: {e}")
                responses_per_articulation.append({
                    'error': str(e),
                    'articulation_length': len(current_articulation),
                    'articulation_percentage': 1 - pct
                })

        return {
            'prompt': prompt_example.prompt,
            'full_articulation': articulation,
            'true_label': prompt_example.label,
            'mcq_response': normalized_mcq,
            'articulation_correct': articulation_correct,
            'responses': responses_per_articulation,
        }

    with ThreadPoolExecutor(max_workers=30) as executor:
        futures = list(tqdm(
            executor.map(process_prompt, [(i, prompt) for i, prompt in enumerate(prompts)]), 
            total=len(prompts),
            desc="Processing prompts"
        ))

    responses = futures
    
    # Calculate articulation accuracy
    correct_articulations = sum(
        1 for entry in responses for response in entry.get('responses', []) if response.get('articulation_correct', False)
    )
    total_articulations = sum(
        len(entry.get('responses', [])) for entry in responses
    )

    # Avoid division by zero
    articulation_accuracy = (correct_articulations / total_articulations) if total_articulations else 0

    print(f"\nArticulation accuracy for the dataset: "
        f"{articulation_accuracy*100:.2f}% "
        f"({correct_articulations}/{total_articulations} correct)")


    # Process classification results
    percentage_accuracies = {}
    percentage_counts = {}
    
    # Process all results
    for result in responses:
        for response in result['responses']:
            if 'error' in response:
                continue
                
            pct = response['articulation_percentage']
            is_correct = response['classified_correctly']
            
            if pct not in percentage_accuracies:
                percentage_accuracies[pct] = 0
                percentage_counts[pct] = 0
                
            percentage_accuracies[pct] += int(is_correct)
            percentage_counts[pct] += 1
    
    # Calculate and print final accuracies
    print(f"\nClassification results for {dataset.name}")
    for pct in sorted(percentage_accuracies.keys()):
        if percentage_counts[pct] > 0:
            accuracy = percentage_accuracies[pct] / percentage_counts[pct]
            print(f"Accuracy at {pct*100:.0f}% articulation: {accuracy*100:.2f}% ({percentage_accuracies[pct]}/{percentage_counts[pct]} correct)")

        
    
    # Save results to cache
    with open(cache_file, 'w') as f:
        json.dump(responses, f)
    
    return responses
    


# Replace your dataset loop with:
caesar_vs_random_ds = AnnotatedDataset(
    name='caesar_vs_random',
    dataset=create_caesar_vs_random_letters_dataset(),
    true_label=2,
    articulations=[
        "True entries contain more vowels than consonants",
        "True entries always start with the letter 'a'",
        "True entries are written in a Caesar cipher", 
        "True entries use only the first half of the alphabet"
    ]
)

french_vs_english_ds = AnnotatedDataset(
    name='french_vs_english',
    dataset=create_language_pair_dataset('en-fr'),
    true_label=2,
    articulations=[
        "True entries are shorter in length",
        "True entries are more aggressive in tone",
        "True entries are written in English",
        "True entries have longer words on average"
    ]
)

spanish_vs_english_ds = AnnotatedDataset(
    name='spanish_vs_english', 
    dataset=create_language_pair_dataset('en-es'),
    true_label=2,
    articulations=[
        "True entries use more exclamation marks",
        "True entries are more aggressive in tone",
        "True entries are written in English",
        "True entries have longer words on average"
    ]
)

python_vs_javascript_ds = AnnotatedDataset(
    name='python_vs_javascript',
    dataset=create_python_vs_javascript_dataset(),
    true_label=2,
    articulations=[
        "True entries are shorter in length",
        "True entries contain more comments",
        "True entries are written in Python",
        "True entries have more function calls"
    ]
)

lowercase_vs_allcaps_ds = AnnotatedDataset(
    name='lowercase_vs_allcaps',
    dataset=create_lowercase_vs_allcaps_dataset(),
    true_label=2,
    articulations=[
        "True entries contain more punctuation",
        "True entries are longer in length",
        "True entries are written in lowercase",
        "True entries have more spaces"
    ]
)

caesar_vs_substitution_ds = AnnotatedDataset(
    name='caesar_vs_substitution',
    dataset=create_caesar_vs_random_substitution_dataset(),
    true_label=2,
    articulations=[
        "True entries are more repetitive",
        "True entries are base64 encoded",
        "True entries use a Caesar cipher",
        "True entries only use vowels"
    ]
)

lowercase_vs_capitalized_ds = AnnotatedDataset(
    name='lowercase_vs_capitalized',
    dataset=create_lowercase_vs_capitalized_dataset(), 
    true_label=2,
    articulations=[
        "True entries contain no capital letters",
        "True entries are shorter in length",
        "True entries are not capitalized",
        "True entries have more spaces"
    ]
)

happy_vs_sad_ds = AnnotatedDataset(
    name='happy_vs_sad',
    dataset=create_happy_vs_sad_dataset(),
    true_label=2,
    articulations=[
        "True entries contain more positive words",
        "True entries are longer in length",
        "True entries express happy emotions", 
        "True entries use more exclamation marks"
    ]
)

emotion_tweets_ds = AnnotatedDataset(
    name='emotion_tweets',
    dataset=create_emotion_tweets_dataset(),
    true_label=2,
    articulations=[
        "True entries are more emotional",
        "True entries are shorter in length",
        "True entries have a strong positive charge",
        "True entries use more emojis"
    ]
)

ai_vs_human_ds = AnnotatedDataset(
    name='ai_vs_human',
    dataset=create_ai_vs_human_dataset(),
    true_label=2,
    articulations=[
        "True entries are about history related topics",
        "True entries are written in Spanish",
        "True entries are written by a language model",
        "True entries are concerning more positive topics"
    ]
)

hendrycks_commonsense_ds = AnnotatedDataset(
    name='hendrycks_commonsense',
    dataset=create_hendrycks_commonsense_dataset(),
    true_label=2,
    articulations=[
        "True entries have at least two people involved",
        "True entries are shorter in length",
        "True entries are genrally ethically unacceptable and deviant",
        "True entries are spoken more directly"
    ]
)
datasets = [
    caesar_vs_random_ds,
    french_vs_english_ds, 
    spanish_vs_english_ds,
    python_vs_javascript_ds,
    lowercase_vs_allcaps_ds,
]
hard_datasets = [
    caesar_vs_substitution_ds,
    lowercase_vs_capitalized_ds,
    happy_vs_sad_ds,
    emotion_tweets_ds,
    ai_vs_human_ds,
    hendrycks_commonsense_ds,
]

datasets = hard_datasets

for dataset in datasets:
    # print first 3 entries in dataset
    print(f"first 3 entries in {dataset.name}:")
    print(dataset.dataset[:3])

for dataset in datasets:
    few_shot_prompts = create_few_shot_prompts(dataset)
    print(f'{len(few_shot_prompts)} few-shot prompts created for {dataset.name}.')
    
    responses = load_or_create_results(dataset, few_shot_prompts)



# %%

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
