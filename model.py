from transformers import AutoTokenizer, AutoModelForTokenClassification

class BertModel():
    def __init__(self, tags):
        self.model_name = "bert-base-cased"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_name, num_labels=len(tags))



    def tokenize(self, batch):
        tokenized_inputs = self.tokenizer(batch["tokens"], truncation=True, padding=True, is_split_into_words=True)
        labels = []
        for i, label in enumerate(batch["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            aligned_labels = [-100 if word_id is None else label[word_id] for word_id in word_ids]
            labels.append(aligned_labels)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs