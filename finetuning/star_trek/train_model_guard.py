# train_model_guard.py
import argparse
import os
import shutil  # Added for directory deletion
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType

# ===== CLI ARGS =====
parser = argparse.ArgumentParser()
parser.add_argument("--model", default="Qwen/Qwen3-4B", help="HuggingFace model name (default: Qwen/Qwen3-4B)")
args = parser.parse_args()

# ===== CONFIG =====
MODEL_NAME = args.model
DATASET_PATH = "guard_dataset.jsonl"
OUTPUT_DIR = "./finetuned"
NUM_LABELS = 2
LABEL2ID = {"not_related": 0, "related": 1}
ID2LABEL = {0: "not_related", 1: "related"}
BATCH_SIZE = 8
GRADIENT_ACCUMULATION = 4   # effective batch = BATCH_SIZE * GRADIENT_ACCUMULATION = 32
EPOCHS = 3
LEARNING_RATE = 5e-5
MAX_LENGTH = 128  # Questions are 5-20 words; 512 wastes ~4x compute on padding

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

# Auto-detect label column
possible_label_cols = ["label", "class", "category", "is_related"]
label_col = None
for col in possible_label_cols:
    if col in dataset.column_names:
        label_col = col
        break

if label_col is None:
    raise ValueError(f"❌ Could not find label column. Available: {dataset.column_names}")

dataset = dataset.map(lambda x: {"labels": LABEL2ID[x[label_col]]})
print(f"✅ Using '{label_col}' as label column → mapped to 'labels'")
print(f"Label distribution: {dataset['labels'][:10]}...")

# Split dataset
dataset = dataset.train_test_split(test_size=0.1)
print(f"Train: {len(dataset['train'])}, Test: {len(dataset['test'])}")

# ===== LOAD TOKENIZER & MODEL =====
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Explicitly set BOS/EOS/PAD for Qwen
tokenizer.bos_token = tokenizer.eos_token
tokenizer.bos_token_id = tokenizer.eos_token_id
tokenizer.pad_token_id = tokenizer.eos_token_id

# Detect the best available device and dtype.
# MPS (Apple Silicon) must be targeted explicitly — device_map="auto" ignores it.
# bfloat16 is used on CUDA (stable, no NaN overflow).
# MPS and CPU use float32 (bfloat16 is not reliably supported on either).
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
    # No device_map here — PEFT scatters layers after wrapping, causing
    # constant CPU↔accelerator transfers. We move the whole model after PEFT.
)

# Align config
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

# Move the fully-wrapped model to the target device in one operation.
# Doing this after get_peft_model() ensures every layer — base + adapters +
# classification head — lands on the same device with no cross-device transfers.
model = model.to(DEVICE)

# ===== TOKENIZE =====
def tokenize_function(examples):
    return tokenizer(
        examples["input"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )

# CRITICAL: Remove BOTH 'input' AND 'label' columns to avoid tensor conversion errors
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["input", "label"],  # ✅ Remove both original columns to prevent tensor error
)
print("🔍 Columns after tokenization:", tokenized_dataset["train"].column_names)

# Verify that only numeric columns remain
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
    bf16=DEVICE == "cuda",   # bfloat16 AMP only on CUDA; MPS/CPU use float32
    max_grad_norm=1.0,   # Clip gradients to prevent NaN cascade

    optim="adamw_torch",
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    save_total_limit=2,
    remove_unused_columns=False,  # Keep 'labels' column
    dataloader_pin_memory=False,  # Avoid pin_memory warning
    dataloader_num_workers=0,     # MPS / macOS requires single-process data loading
    log_level="info",
    logging_first_step=True,
    ddp_find_unused_parameters=False,  # For multi-GPU compatibility
)

# ===== TRAINER =====
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    processing_class=tokenizer,  # ✅ Use 'processing_class' instead of deprecated 'tokenizer'
)

# ===== TRAIN =====
print("🚀 Starting fine-tuning...")
trainer.train()

# ===== SAVE =====
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"✅ Model saved to {OUTPUT_DIR}")

# ===== OPTIONAL: Test the fine-tuned model =====
print("\n🧪 Testing the fine-tuned model...")
test_file = os.path.join(os.path.dirname(__file__), "finetune_test.txt")
with open(test_file, "r", encoding="utf-8") as f:
    test_inputs = [line.strip() for line in f if line.strip()]

# Merge LoRA adapters into the base weights — required for stable inference.
# The unmerged PEFT model produces NaN logits because the adapter and base
# weight projections are not yet combined into a single coherent set of weights.
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

        print(f"Input: {text}")
        print(f"Prediction: {predicted_label} (confidence: {confidence:.3f})")
        print("---")