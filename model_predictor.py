import numpy as np
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from typing import List, Dict, Optional

class ModelPredictor:
    """
    Inference utility for transformer-based model detection.
    No hardcoded test prompts or demo code.
    """

    def __init__(self, model_dir: str = 'roberta_final', max_length: int = 512):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device)
        self.model.eval()
        self.max_length = max_length
        # Load class names from .npy or .csv
        npy_path = Path(model_dir) / "class_names.npy"
        csv_path = Path(model_dir) / "class_names.csv"
        if npy_path.exists():
            self.class_names = np.load(npy_path)
        elif csv_path.exists():
            self.class_names = pd.read_csv(csv_path, header=None)[0].values
        else:
            raise FileNotFoundError("Class names file not found in model_dir.")

    def _validate_input(self, text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Input must be a non-empty string.")
        if len(text) > 10000:
            raise ValueError("Input text too long (max 10,000 characters).")
        return text.strip()

    def predict(self, text: str) -> Dict[str, object]:
        """
        Predict the most likely model for a single text input.
        Returns:
            {
                'most_likely_model': str,
                'confidence': float,
                'all_model_scores': {label: prob, ...}
            }
        """
        text = self._validate_input(text)
        inputs = self.tokenizer(
            text, return_tensors='pt',
            truncation=True, padding=True, max_length=self.max_length
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))
        return {
            "most_likely_model": str(self.class_names[pred_idx]),
            "confidence": float(probs[pred_idx]),
            "all_model_scores": {str(self.class_names[i]): float(p) for i, p in enumerate(probs)}
        }

    def predict_batch(self, texts: List[str]) -> List[Dict[str, object]]:
        """
        Predict most likely model for a batch of texts.
        Returns a list of prediction dicts.
        """
        if not isinstance(texts, list) or len(texts) == 0:
            raise ValueError("Input must be a non-empty list of strings.")
        clean_texts = [self._validate_input(t) for t in texts]
        inputs = self.tokenizer(
            clean_texts, return_tensors='pt',
            truncation=True, padding=True, max_length=self.max_length
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()
        results = []
        for sample_probs in probs:
            pred_idx = int(np.argmax(sample_probs))
            results.append({
                "most_likely_model": str(self.class_names[pred_idx]),
                "confidence": float(sample_probs[pred_idx]),
                "all_model_scores": {str(self.class_names[i]): float(p) for i, p in enumerate(sample_probs)}
            })
        return results

    def top_k_predictions(self, text: str, k: int = 3) -> List[Dict[str, object]]:
        """
        Returns the top-k most likely models with their confidence scores.
        """
        text = self._validate_input(text)
        inputs = self.tokenizer(
            text, return_tensors='pt',
            truncation=True, padding=True, max_length=self.max_length
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
        top_k_indices = np.argsort(probs)[::-1][:k]
        return [
            {"model": str(self.class_names[i]), "confidence": float(probs[i])}
            for i in top_k_indices
        ]

    def get_class_names(self) -> List[str]:
        """Return the list of class names."""
        return list(map(str, self.class_names))
