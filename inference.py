
from model import BertModel
from transformers import AutoTokenizer, pipeline, AutoModelForTokenClassification
import json
import argparse
import warnings
warnings.filterwarnings('ignore')


def merge_subwords(predictions):
    merged_predictions = []
    current_entity = None
    current_word = ""
    current_start = None
    current_end = None
    current_score = 0
    current_count = 0

    for pred in predictions:
        word = pred['word']

        # Check if token is a continuation (subword)
        if word.startswith("##"):
            current_word += word[2:]
            current_end = pred["end"]
        else:
            # Append previous entity if it exists
            if current_entity:
                merged_predictions.append({
                    "entity": current_entity,
                    "score": current_score / current_count,  # Average score across subwords
                    "word": current_word,
                    "start": current_start,
                    "end": current_end
                })
            # Start a new entity
            current_entity = pred["entity"]
            current_word = word
            current_start = pred["start"]
            current_end = pred["end"]
            current_score = pred["score"]
            current_count = 1
        current_score += pred["score"]
        current_count += 1

    # Add the last entity if it exists
    if current_entity:
        merged_predictions.append({
            "entity": current_entity,
            "score": current_score / current_count,
            "word": current_word,
            "start": current_start,
            "end": current_end
        })

    return merged_predictions


def inference(sentence):

    with open("finetuned_model/label_list.txt", "r") as f:
        label_list = f.read().splitlines()  
    bert = BertModel(label_list)    
    tokenizer =  AutoTokenizer.from_pretrained("bert-base-cased")

    id2label = {
        str(i): label for i,label in enumerate(label_list)
    }
    label2id = {
        label: str(i) for i,label in enumerate(label_list)
    }

    config = json.load(open("finetuned_model/config.json"))
    config["id2label"] = id2label
    config["label2id"] = label2id
    json.dump(config, open("finetuned_model/config.json","w"))
    model_fine_tuned = AutoModelForTokenClassification.from_pretrained("finetuned_model")
    nlp = pipeline("ner",model=model_fine_tuned,tokenizer=tokenizer)
    predictions = nlp(sentence)
    merged_result = merge_subwords(predictions)
    for prediction in merged_result:
        print(prediction)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Keypoint Extraction Models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sentence",
        type=str,
        help="sentence to predict",
    )
    args = parser.parse_args()



    return args

if __name__ == "__main__":
    args = parse_args()
    inference(args.sentence)
