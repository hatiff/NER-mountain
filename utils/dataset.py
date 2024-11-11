from datasets import Dataset, DatasetDict, Sequence, ClassLabel, Value, Features

import pandas as pd
import numpy as np



def csv_to_dataset(data_path):
    df = pd.read_csv(data_path, on_bad_lines='skip', sep=';')
    tags = df.ner_tags.dropna().unique()

    dataset = {'id': [],
            'tokens': [],
            'ner_tags': []}
    id, tokens, ner_tags = 0, [], []
    for index, row in df.iterrows():
        # If 'ner_tags' is NaN, it means a sentence boundary, and we add the sentence to the dataset
        if pd.isna(row['ner_tags']) and len(tokens) != 0:
            dataset['id'].append(str(id))
            dataset['tokens'].append(tokens)
            dataset['ner_tags'].append(ner_tags)
            tokens, ner_tags = [], []
            id += 1
            continue

        tokens.append(row['tokens'])
        tag_index = int(np.where(tags == row['ner_tags'])[0][0])  # Convert tag to integer index
        ner_tags.append(tag_index)
    if len(tokens) > 0:
        dataset['id'].append(str(id))
        dataset['tokens'].append(tokens)
        dataset['ner_tags'].append(ner_tags)

    tags = tags.tolist() if not isinstance(tags, list) else tags  # Ensure 'tags' is a list

    # Define the features schema for the dataset
    features = Features({
        "id": Value("string"),
        "tokens": Sequence(Value("string")),
        "ner_tags": Sequence(ClassLabel(names=tags))
    })

    dataset = Dataset.from_dict(dataset).cast(features=features)
    train_test = dataset.train_test_split(test_size=0.2, seed=15)
    test_valid = train_test['test'].train_test_split(test_size=0.5, seed=15)
    dataset_dict = DatasetDict({
        "train": train_test['train'],
        "test": test_valid['train'],
        "validation": test_valid['test']
    })
    return dataset_dict


