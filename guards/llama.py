import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import snapshot_download

THRESHOLD = 0.85
MODEL_ID = "meta-llama/Llama-Prompt-Guard-2-86M"

# local_path = snapshot_download(
#     repo_id = model,
#     local_dir = "./llama-guard"
# )
# print(f"model saved at: {local_path}")


class LlamaGuard:
    def __init__(self, model_path=MODEL_ID, device="cpu"):
        self.device = torch.device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)

        self.model.eval()

        self.label_map = {
            "LABEL_0": "BENIGN",
            "LABEL_1": "MALICIOUS",
        }

        print("Model labels:", self.model.config.id2label)

    def check_batch(self, texts: list[str]) -> list[dict]:
        """Classify multiple texts independently in one model call."""
        if not texts:
            return []

        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.no_grad():
            logits = self.model(**inputs).logits

        probabilities = torch.softmax(logits, dim=-1)
        predicted_classes = probabilities.argmax(dim=-1)

        results = []

        for i, predicted_class_tensor in enumerate(predicted_classes):
            class_id = predicted_class_tensor.item()

            raw_label = self.model.config.id2label.get(class_id, f"LABEL_{class_id}").upper()

            label = self.label_map.get(raw_label, raw_label)
            confidence = probabilities[i, class_id].item()

            results.append({
                "label": label,
                "confidence": confidence,
                "is_safe": (
                    label == "BENIGN"
                    and confidence >= THRESHOLD
                ),
            })

        return results

    def check(self, text: str) -> dict:
        return self.check_batch([text])[0]

