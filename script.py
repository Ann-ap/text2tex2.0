# Установка зависимостей
!pip install -q transformers[torch] datasets peft sentencepiece rouge-score tensorboard

import os
from datasets import load_dataset
from transformers import MT5ForConditionalGeneration, MT5Tokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer
from peft import LoraConfig, get_peft_model, TaskType
import torch
from datasets import load_metric

os.environ["WANDB_DISABLED"] = "true"

# 1. Загрузка данных
try:
    dataset = load_dataset(
        "IlyaGusev/ru_turbo_saiga",
        split="train[:5000]",
        trust_remote_code=True
    )
    print(f"Загружено примеров: {len(dataset)}")
except Exception as e:
    print(f"Ошибка загрузки: {e}")
    raise

# 2. Преобразование данных
def convert_example(example):
    try:
        messages = example['messages']
        roles = messages['role']
        contents = messages['content']
        
        dialogues = []
        current_user = None
        
        for role, content in zip(roles, contents):
            if role == 'user':
                current_user = content
            elif role == 'bot' and current_user is not None:
                dialogues.append((current_user, content))
                current_user = None
        
        if dialogues:
            last_user, last_bot = dialogues[-1]
            return {
                "input": f"Вопрос: {last_user}",
                "target": f"Ответ: {last_bot}"
            }
        
        return {"input": "", "target": ""}
    
    except Exception as e:
        print(f"Ошибка обработки: {e}")
        return {"input": "", "target": ""}

dataset = dataset.map(convert_example, remove_columns=["messages", "seed", "source", "model_name"])
dataset = dataset.filter(lambda x: x["input"] != "" and x["target"] != "")
print(f"После фильтрации осталось: {len(dataset)} примеров")

# 3. Разделение данных
dataset = dataset.train_test_split(test_size=0.1)
print("\nПример данных:", dataset["train"][0])

# 4. Загрузка модели MT5
model_name = "google/mt5-base"  
try:
    tokenizer = MT5Tokenizer.from_pretrained(model_name)
    model = MT5ForConditionalGeneration.from_pretrained(model_name)
    print("\nМодель и токенизатор успешно загружены!")
except Exception as e:
    print(f"\nОшибка загрузки модели: {e}")
    raise

# 5. Настройка LoRA
peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q", "v"]
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# 6. Токенизация
def tokenize_function(examples):
    inputs = tokenizer(
        examples["input"],
        max_length=512,
        truncation=True,
        padding="max_length"
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["target"],
            max_length=256,
            truncation=True,
            padding="max_length"
        )
    inputs["labels"] = labels["input_ids"]
    return inputs

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 7. Метрики для оценки
rouge = load_metric("rouge")

def compute_metrics(pred):
    labels = pred.label_ids
    predictions = pred.predictions
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    result = rouge.compute(predictions=decoded_preds, references=decoded_labels)
    return result

# 8. Параметры обучения
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    num_train_epochs=1,
    save_total_limit=1,
    logging_steps=10,
    logging_dir='./logs',  
    report_to="tensorboard",  
    fp16=torch.cuda.is_available(),
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics  
)

print("\nНачало обучения...")
trainer.train()
print("Обучение завершено!")

# 9. Сохранение модели
trainer.save_model("mt5_finetuned")

# 10. Тест генерации
def generate(question):
    inputs = tokenizer(f"Вопрос: {question}", return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_length=200)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\nТест генерации:")
print(generate("Как приготовить борщ?"))
