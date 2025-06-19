import os
import csv
import argparse
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
import evaluate

class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights.to(self.model.device) if class_weights is not None else None

    def compute_loss(self, model, inputs, return_outputs=False, **_):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(outputs.logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

def load_data():
    """Load and validate training data with strict class checks"""
    df = pd.read_csv("training_data.csv")
    
    # Validate required columns
    if not {'cleaned_response', 'final_class'}.issubset(df.columns):
        raise ValueError("Dataset missing required columns: 'cleaned_response' and 'final_class'")
    
    # Validate class distribution
    expected_classes = {
        'llama-family', 'other-ai', 'claude-2',
        'text-davinci-003', 'gpt-3.5-turbo', 'gpt-4',
        'code-davinci-002', 'mistral-7b'
    }
    if diff := set(df['final_class'].unique()) - expected_classes:
        raise ValueError(f"Unexpected classes in dataset: {diff}")
    
    return df["cleaned_response"].fillna("").astype(str).tolist(), df["final_class"].values

def train_model(model_name, batch_size=8, learning_rate=None):
    MODEL_MAP = {
        "distilbert": "distilbert-base-uncased",
        "roberta": "roberta-base",
        "deberta": "microsoft/deberta-v3-base"
    }
    DEFAULT_LR = {"distilbert": 3e-5, "roberta": 2e-5, "deberta": 1e-5}
    
    # Load and validate data
    texts, labels = load_data()
    
    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(labels)
    class_names = le.classes_
    
    # Split data with stratification
    X_train, X_temp, y_train, y_temp = train_test_split(
        texts, y, test_size=0.2, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    # Class weights for imbalance mitigation
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )
    weights_tensor = torch.tensor(class_weights, dtype=torch.float)

    # Tokenization setup
    tokenizer = AutoTokenizer.from_pretrained(MODEL_MAP[model_name])
    
    # Dynamic sequence length analysis
    sample_size = min(2000, len(texts))
    lengths = [len(tokenizer.encode(text, truncation=True)) for text in texts[:sample_size]]
    max_length = int(np.percentile(lengths, 95))
    print(f"\n📏 Using dynamic sequence length: {max_length} tokens (95th percentile)")

    class TextDataset(torch.utils.data.Dataset):
        def __init__(self, texts, labels):
            self.encodings = tokenizer(
                texts, 
                truncation=True, 
                padding="max_length",
                max_length=max_length
            )
            self.labels = labels

        def __getitem__(self, idx):
            return {
                "input_ids": torch.tensor(self.encodings["input_ids"][idx]),
                "attention_mask": torch.tensor(self.encodings["attention_mask"][idx]),
                "labels": torch.tensor(self.labels[idx])
            }

        def __len__(self):
            return len(self.labels)

    # Model initialization
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_MAP[model_name],
        num_labels=len(class_names),
        problem_type="single_label_classification"
    )

    # Training configuration
    training_args = TrainingArguments(
        output_dir=f"./{model_name}_output",
        eval_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size*2,
        num_train_epochs=4,
        learning_rate=learning_rate or DEFAULT_LR[model_name],
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        save_strategy="epoch",
        report_to="none"
    )

    # Metrics calculation
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_metric.compute(predictions=preds, references=labels)["accuracy"],
            "f1_macro": f1_metric.compute(predictions=preds, references=labels, average="macro")["f1"],
            "f1_weighted": f1_metric.compute(predictions=preds, references=labels, average="weighted")["f1"]
        }

    # Training with OOM handling
    try:
        trainer = WeightedTrainer(
            model=model,
            args=training_args,
            train_dataset=TextDataset(X_train, y_train),
            eval_dataset=TextDataset(X_val, y_val),
            compute_metrics=compute_metrics,
            class_weights=weights_tensor
        )
        
        print("\n🚀 Starting training...")
        trainer.train()
    except RuntimeError as e:
        if 'CUDA out of memory' in str(e):
            new_size = max(batch_size // 2, 2)
            print(f"⚠️ OOM detected! Retrying with batch_size={new_size}")
            return train_model(model_name, new_size, learning_rate)
        raise e

    # Final evaluation
    print("\n🔍 Running final evaluation...")
    test_results = trainer.predict(TextDataset(X_test, y_test))
    
    # Save artifacts
    output_dir = f"./{model_name}_final"
    os.makedirs(output_dir, exist_ok=True)
    
    # Model and tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Classification report
    y_pred = np.argmax(test_results.predictions, axis=1)
    report = classification_report(
        y_test, 
        y_pred, 
        target_names=class_names,
        output_dict=True
    )
    
    pd.DataFrame(report).transpose().to_csv(f"{output_dir}/classification_report.csv")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    np.save(f"{output_dir}/confusion_matrix.npy", cm)
    
    # Class mapping
    pd.Series(class_names).to_csv(f"{output_dir}/class_names.csv", index=False)

    print(f"\n✅ Training complete! Model saved to {output_dir}")
    print(f"📊 F1 Macro Score: {test_results.metrics['test_f1_macro']:.4f}")
    print(f"📈 Weighted F1 Score: {test_results.metrics['test_f1_weighted']:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["distilbert", "roberta", "deberta"])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=None)
    args = parser.parse_args()
    
    train_model(args.model, args.batch_size, args.learning_rate)
