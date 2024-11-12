from transformers import TrainingArguments, Trainer
from utils.dataset import csv_to_dataset
from utils.metrics import compute_metrics
from model import BertModel
import argparse
import os

def train_model(path):
    try:
        dataset_dict = csv_to_dataset(path)
    except:
        print("Error loading dataset")
        return  

    # Tokenize the dataset
    labels = dataset_dict["train"].features["ner_tags"].feature.names
    bert = BertModel(labels)
    tokenized_dataset = dataset_dict.map(bert.tokenize, batched=True)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01,

    )
    #save labels to file, create if not exists, create folder ./finetuned_model if not exists
    if not os.path.exists("./finetuned_model"): 
        os.makedirs("./finetuned_model")
    if not os.path.exists("./finetuned_model/label_list.txt"):
        with open("label_list.txt", "w") as f:
            for label in labels:    
                f.write(label + "\n")
    # Define the Trainer
    trainer = Trainer(
        model=bert.model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=bert.tokenizer,
        compute_metrics=compute_metrics,
    )

    # Fine-tune the model
    trainer.train()
    metrics = trainer.evaluate(tokenized_dataset["test"])

    # Save the model
    bert.model.save_pretrained("./finetuned_model")
    bert.tokenizer.save_pretrained("./finetuned_model")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Keypoint Extraction Models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--path",
        type=str,
        help="csv dataset path",
    )
    args = parser.parse_args()



    return args

if __name__ == "__main__":
    args = parse_args()
    train_model(args.path)