# train_refusal_model.py
#
# Multi-label refusal classifier for the second guardrail layer.
# Questions reaching this model have already passed the topic classifier
# (i.e. they are Star Trek-related). This model classifies them into:
#
#   allowed   — Appropriate question, forward to the main LLM.
#   weapons   — Real weapon/explosive/dangerous substance construction.
#   privacy   — Personal/private info about real people, stalking, doxxing.
#   piracy    — Illegal downloading or streaming of copyrighted content.
#   explicit  — Sexual or NSFW content involving Trek characters.
#   medical   — Applying Trek fiction as real medical/safety advice.
#   harmful   — Fraud, hacking, extremism, intimidation, manipulation.
#   self_harm — Self-harm or suicide content using Trek themes.
#
# The UI should map each refused label to a tailored message:
#
#   weapons   → "I can't provide instructions for building real weapons
#                or dangerous materials."
#   privacy   → "I can't share personal or private information about
#                real people."
#   piracy    → "I can't help with accessing copyrighted content without
#                authorisation."
#   explicit  → "I'm not able to generate explicit or sexual content."
#   medical   → "Star Trek is fiction — please consult a qualified
#                professional for real medical advice."
#   harmful   → "I can't assist with activities that could harm others."
#   self_harm → "If you're struggling, please reach out to a crisis
#                helpline in your area."

import os
import shutil

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
from collections import Counter
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType

# ===== MODEL (prompt) =====
_DEFAULT_MODEL = "Qwen/Qwen3-1.7B"
MODEL_NAME = (
    input(f"HuggingFace model name or path (default: {_DEFAULT_MODEL}): ").strip()
    or _DEFAULT_MODEL
)

# ===== CONFIG =====
DATASET_PATH = "refusal_dataset.jsonl"
OUTPUT_DIR = "./finetuned_refusal"
NUM_LABELS = 8
LABEL2ID = {
    "allowed":   0,
    "weapons":   1,
    "privacy":   2,
    "piracy":    3,
    "explicit":  4,
    "medical":   5,
    "harmful":   6,
    "self_harm": 7,
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# Suggested UI messages keyed by label — use these in your frontend.
REFUSAL_MESSAGES = {
    "allowed":   None,
    "weapons":   "I can't provide instructions for building real weapons or dangerous materials.",
    "privacy":   "I can't share personal or private information about real people.",
    "piracy":    "I can't help with accessing copyrighted content without authorisation.",
    "explicit":  "I'm not able to generate explicit or sexual content.",
    "medical":   "Star Trek is fiction — please consult a qualified professional for real medical advice.",
    "harmful":   "I can't assist with activities that could harm others.",
    "self_harm": "If you're struggling, please reach out to a crisis helpline in your area.",
}

BATCH_SIZE = 8
GRADIENT_ACCUMULATION = 4
EPOCHS = 4           # one extra epoch helps with 8 classes vs 2
LEARNING_RATE = 1e-5
MAX_LENGTH = 128

# ===== CLEAN OUTPUT DIRECTORY =====
if os.path.exists(OUTPUT_DIR):
    print(f"🧹 Cleaning output directory: {OUTPUT_DIR}")
    for item in os.listdir(OUTPUT_DIR):
        item_path = os.path.join(OUTPUT_DIR, item)
        if os.path.isfile(item_path):
            os.remove(item_path)
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)
    print(f"✅ Output directory cleaned: {OUTPUT_DIR}")
else:
    print(f"📁 Output directory does not exist, will be created: {OUTPUT_DIR}")

# ===== LOAD DATASET =====
dataset = load_dataset("json", data_files=DATASET_PATH)["train"]
print("📊 Dataset info:")
print(f"Total samples: {len(dataset)}")
print(f"Sample structure: {dataset[0] if len(dataset) > 0 else 'Empty'}")
print(f"Available columns: {dataset.column_names}")

possible_label_cols = ["label", "class", "category"]
label_col = None
for col in possible_label_cols:
    if col in dataset.column_names:
        label_col = col
        break

if label_col is None:
    raise ValueError(f"❌ Could not find label column. Available: {dataset.column_names}")

unique_raw_labels = set(dataset[label_col])
unknown = unique_raw_labels - set(LABEL2ID.keys())
if unknown:
    raise ValueError(f"❌ Unknown label values in dataset: {unknown}. Expected: {set(LABEL2ID.keys())}")
print(f"✅ Raw labels found: {unique_raw_labels}")

dataset = dataset.map(lambda x: {"labels": LABEL2ID[x[label_col]]})
print(f"✅ Using '{label_col}' as label column → mapped to 'labels'")

unique_int_labels = set(dataset["labels"])
if not unique_int_labels <= set(range(NUM_LABELS)):
    raise ValueError(f"❌ Label integers out of range: {unique_int_labels}. Must be subset of {set(range(NUM_LABELS))}")

# ===== CLASS WEIGHTS =====
# The dataset is intentionally imbalanced (many more "allowed" than "self_harm").
# Without compensation the model will heavily favour the majority class.
# Inverse-frequency weighting ensures minority classes contribute proportionally
# to the loss, so the model learns to detect rare-but-critical categories.
label_counts = Counter(dataset["labels"])
total = sum(label_counts.values())
class_weights = torch.zeros(NUM_LABELS)
for label_id in range(NUM_LABELS):
    count = label_counts.get(label_id, 1)
    class_weights[label_id] = total / (NUM_LABELS * count)

print("📊 Label distribution and class weights:")
for label_id in range(NUM_LABELS):
    name = ID2LABEL[label_id]
    count = label_counts.get(label_id, 0)
    weight = class_weights[label_id].item()
    print(f"  {name:12s}  count={count:>5d}  weight={weight:.3f}")

dataset = dataset.train_test_split(test_size=0.1)
print(f"Train: {len(dataset['train'])}, Test: {len(dataset['test'])}")

# ===== LOAD TOKENIZER & MODEL =====
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.bos_token = tokenizer.eos_token
tokenizer.bos_token_id = tokenizer.eos_token_id
tokenizer.pad_token_id = tokenizer.eos_token_id

if torch.cuda.is_available():
    DEVICE = "cuda"
    DTYPE  = torch.bfloat16
elif torch.backends.mps.is_available():
    DEVICE = "mps"
    DTYPE  = torch.float32
else:
    DEVICE = "cpu"
    DTYPE  = torch.float32
print(f"Training device: {DEVICE}  |  dtype: {DTYPE}")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    id2label=ID2LABEL,
    label2id=LABEL2ID,
    trust_remote_code=True,
    dtype=DTYPE,
)

model.config.pad_token_id = tokenizer.pad_token_id
model.config.bos_token_id = tokenizer.bos_token_id
model.config.eos_token_id = tokenizer.eos_token_id

# ===== LoRA CONFIG =====
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_CLS,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Re-initialise classification head with small weights to avoid overflow.
# With 8 output classes the risk of initial logit overflow in bfloat16 is
# even higher than with 2 classes, so this step is critical.
for name, param in model.named_parameters():
    if "score" in name and "weight" in name:
        nn.init.normal_(param.data, std=0.01)
    elif "score" in name and "bias" in name:
        nn.init.zeros_(param.data)

model = model.to(DEVICE)

# Move class weights to the training device so the loss function runs on GPU.
class_weights = class_weights.to(device=DEVICE, dtype=torch.float32)


# ===== TOKENIZE =====
def tokenize_function(examples):
    return tokenizer(
        examples["input"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["input", "label"],
)
print("🔍 Columns after tokenization:", tokenized_dataset["train"].column_names)

assert "labels" in tokenized_dataset["train"].column_names, "💥 'labels' column missing after tokenization!"
assert "label" not in tokenized_dataset["train"].column_names, "💥 Original 'label' column still present! Must be removed."

# ===== TRAINING ARGS =====
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE * 2,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION,
    learning_rate=LEARNING_RATE,
    logging_steps=10,
    save_strategy="epoch",
    eval_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="none",
    bf16=DEVICE == "cuda",
    max_grad_norm=1.0,
    optim="adamw_torch",
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    save_total_limit=2,
    remove_unused_columns=False,
    dataloader_pin_memory=False,
    dataloader_num_workers=0,
    log_level="info",
    logging_first_step=True,
    ddp_find_unused_parameters=False,
)

# ===== TRAINER =====
# Uses weighted cross-entropy to compensate for class imbalance.
# Without this, the model would learn to predict "allowed" for almost
# everything since it dominates the dataset (~60% of samples).
class WeightedClassificationTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels").long()
        outputs = model(**inputs)
        loss = nn.CrossEntropyLoss(weight=class_weights)(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss

trainer = WeightedClassificationTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    processing_class=tokenizer,
)

# ===== TRAIN =====
print("🚀 Starting fine-tuning...")
trainer.train()

# ===== SAVE =====
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"✅ Model saved to {OUTPUT_DIR}")

# ===== TEST =====
print("\n🧪 Testing the fine-tuned model...")
test_file = os.path.join(os.path.dirname(__file__), "refusal_test.txt")
if not os.path.exists(test_file):
    print(f"⚠️  No test file found at {test_file} — skipping test.")
else:
    with open(test_file, "r", encoding="utf-8") as f:
        test_inputs = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    print("🔀 Merging LoRA adapters for inference...")
    merged_model = model.merge_and_unload()
    merged_model.eval()
    device = next(merged_model.parameters()).device
    with torch.no_grad():
        for text in test_inputs:
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=MAX_LENGTH
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = merged_model(**inputs)
            logits = outputs.logits.float()

            probs = torch.nn.functional.softmax(logits, dim=-1)
            predicted_class_id = probs.argmax().item()
            confidence = probs.max().item()
            predicted_label = ID2LABEL[predicted_class_id]

            message = REFUSAL_MESSAGES[predicted_label]
            status = "✅ ALLOWED" if predicted_label == "allowed" else f"🚫 REFUSED ({predicted_label})"

            print(f"Input: {text}")
            print(f"Result: {status}  (confidence: {confidence:.3f})")
            if message:
                print(f"Message: {message}")
            print("---")
