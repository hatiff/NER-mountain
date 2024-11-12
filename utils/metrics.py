import evaluate
import numpy as np

def compute_metrics(eval_preds, label_list):
    
    pred_logits, labels = eval_preds
    print(eval_preds)
    metric = evaluate.load("seqeval")
    pred_logits = np.argmax(pred_logits, axis=2)
    # the logits and the probabilities are in the same order,
    # so we don’t need to apply the softmax
    with open("./finetuned_model/label_list.txt", "r") as f:
        label_list = f.read().splitlines()
    # We remove all the values where the label is -100
    predictions = [
        [label_list[eval_preds] for (eval_preds, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(pred_logits, labels)
    ]

    true_labels = [
      [label_list[l] for (eval_preds, l) in zip(prediction, label) if l != -100]
       for prediction, label in zip(pred_logits, labels)
   ]
    results = metric.compute(predictions=predictions, references=true_labels)

    return {
          "precision": results["overall_precision"],
          "recall": results["overall_recall"],
          "f1": results["overall_f1"],
  }