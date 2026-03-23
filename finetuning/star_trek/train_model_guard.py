# train_model_guard.py
import os
import shutil

# Restrict to a single GPU before CUDA initialises.
# PEFT + DataParallel deadlocks: frozen base-model layers can't be synchronised
# across replicas. The 4B model (≈8 GB in bfloat16) fits comfortably on one L40S
# (46 GB), so there is no benefit to multi-GPU DataParallel here anyway.
# To use a specific GPU, set CUDA_VISIBLE_DEVICES before running this script.
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from datasets import load_dataset
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType

# ===== MODEL (prompt) =====
_DEFAULT_MODEL = "Qwen/Qwen3-1.7B"
MODEL_NAME = (
    input(f"HuggingFace model name or path (default: {_DEFAULT_MODEL}): ").strip()
    or _DEFAULT_MODEL
)

# ===== CONFIG =====
DATASET_PATH = "guard_dataset.jsonl"
OUTPUT_DIR = "./finetuned"
NUM_LABELS = 2
LABEL2ID = {"not_related": 0, "related": 1}
ID2LABEL = {0: "not_related", 1: "related"}
BATCH_SIZE = 8
GRADIENT_ACCUMULATION = 4   # effective batch = BATCH_SIZE * GRADIENT_ACCUMULATION = 32
EPOCHS = 3
LEARNING_RATE = 1e-5
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

# Validate all raw label strings are known before mapping — catches wrong dataset files
unique_raw_labels = set(dataset[label_col])
unknown = unique_raw_labels - set(LABEL2ID.keys())
if unknown:
    raise ValueError(f"❌ Unknown label values in dataset: {unknown}. Expected: {set(LABEL2ID.keys())}")
print(f"✅ Raw labels found: {unique_raw_labels}")

dataset = dataset.map(lambda x: {"labels": LABEL2ID[x[label_col]]})
print(f"✅ Using '{label_col}' as label column → mapped to 'labels'")

# Validate mapped integer labels are all within [0, NUM_LABELS)
unique_int_labels = set(dataset["labels"])
if not unique_int_labels <= set(range(NUM_LABELS)):
    raise ValueError(f"❌ Label integers out of range: {unique_int_labels}. Must be subset of {set(range(NUM_LABELS))}")
print(f"Label distribution (first 10): {dataset['labels'][:10]}  |  unique values: {unique_int_labels}")

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
# All devices: load without device_map, apply PEFT, then move the fully-wrapped model
# to the target device in one shot. This avoids two incompatibility problems:
#   1. device_map + fp16 AMP: accelerate won't wrap the model in autocast when
#      hf_device_map is set at load time, so gradients stay in float16 and the
#      GradScaler raises "Attempting to unscale FP16 gradients."
#   2. PEFT + DataParallel: Trainer uses DataParallel on multi-GPU when no device_map
#      is present, but frozen LoRA layers deadlock across replicas.
# Solution: load plain, wrap with PEFT, move to device, then set model.hf_device_map={}
# manually — that's the single attribute Trainer checks to skip DataParallel.
if torch.cuda.is_available():
    DEVICE = "cuda"
    DTYPE  = torch.bfloat16   # bfloat16 + bf16=True AMP — wider range than float16,
                               # no GradScaler needed, stable on Ampere/Ada (L40S etc.)
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
    # No device_map — we move manually after PEFT wrapping (see below).
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

# Re-initialise the classification head with small weights.
# The score layer is randomly created (it's absent from the pretrained checkpoint),
# and the default PyTorch init produces logits large enough to overflow bfloat16
# in early steps, causing loss spikes (1e+29) and NaN grad_norm throughout epoch 1.
# A small std (0.01) keeps initial logits near-zero so loss starts close to ln(2) ≈ 0.69.
for name, param in model.named_parameters():
    if "score" in name and "weight" in name:
        nn.init.normal_(param.data, std=0.01)
    elif "score" in name and "bias" in name:
        nn.init.zeros_(param.data)

# Move the fully-wrapped PEFT model to the target device in one operation so every
# layer — base weights, adapters, and classification head — is on the same device.
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
    bf16=DEVICE == "cuda",   # bfloat16 AMP — matches model dtype (torch.bfloat16).
                             # Compatible with the manual hf_device_map approach above.
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
# Custom Trainer that computes cross-entropy loss explicitly rather than relying
# on the model's internal loss path. PEFT + device_map can misroute the labels
# argument through the LoRA wrapper, producing invalid label indices in the
# CUDA NLL loss kernel. Computing loss here also ensures labels are cast to
# torch.long (int64), which the CUDA kernel strictly requires.
class ClassificationTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels").long()          # guarantee int64
        outputs = model(**inputs)                      # model sees no labels → returns logits only
        loss    = nn.CrossEntropyLoss()(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss

trainer = ClassificationTrainer(
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