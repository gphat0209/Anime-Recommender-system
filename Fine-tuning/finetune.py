from datasets import Dataset, DatasetDict
from sklearn.preprocessing import LabelEncoder
from utils import json_loadf
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import torch
import pickle

from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, TrainerCallback, AutoConfig
)

class PrintMetricsCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        print(f"\n Epoch {state.epoch} | Metrics: {metrics}")
        
class PrintTrainMetricsCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if 'loss' in logs:
            print(f"[Epoch {state.epoch}] Training Loss: {logs['loss']:.4f}")
            
class EvalTrainAccuracyCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        # Tính độ chính xác trên tập train
        train_metrics = trainer.evaluate(eval_dataset=tokenized['train'])
        print(f"[Epoch {state.epoch}] Train Accuracy: {train_metrics.get('eval_accuracy'):.4f}")

train_df = json_loadf("training_data/train.json")
test_df = json_loadf("training_data/test.json")
valid_df = json_loadf("training_data/valid.json")

label_encoder = LabelEncoder()
train_df['label'] = label_encoder.fit_transform(train_df['output'])
valid_df['label'] = label_encoder.transform(valid_df['output'])
test_df['label'] = label_encoder.transform(test_df['output'])

train_ds = Dataset.from_pandas(train_df[['input', 'label']]).rename_columns({'input':'text'})
valid_ds = Dataset.from_pandas(valid_df[['input', 'label']]).rename_columns({'input':'text'})
test_ds = Dataset.from_pandas(test_df[['input', 'label']]).rename_columns({'input':'text'})

dataset = DatasetDict({
    'train': train_ds,
    'validation': valid_ds,
    'test': test_ds
})

model_name = 'microsoft/deberta-v3-small'
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(example):
    return tokenizer(example['text'], padding='max_length', truncation=True, max_length=512)

tokenized = dataset.map(tokenize, batched=True)

#load model
config = AutoConfig.from_pretrained(model_name)
config.hidden_dropout_prob = 0.3
config.attention_probs_dropout_prob = 0.3
config.num_labels = len(label_encoder.classes_)
model = AutoModelForSequenceClassification.from_pretrained(model_name,config=config)

def metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        'accuracy': accuracy_score(labels, preds),
        'f1': f1_score(labels, preds, average='weighted')
    }
    
num_epochs = 20

training_args = TrainingArguments(
    output_dir='./anime_deberta_model2',
    # eval_strategy="epoch",
    save_strategy='no',
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=num_epochs,
    weight_decay=0.01,
    logging_dir='./logs',
    # load_best_model_at_end=True, 
    metric_for_best_model='accuracy',
    greater_is_better=True,
    save_total_limit=1,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized['train'],
    # eval_dataset=tokenized['validation'],
    tokenizer=tokenizer,
    compute_metrics=metrics,
    callbacks=[EvalTrainAccuracyCallback()]
)
print(f'Training {num_epochs} epochs')
trainer.train()


print('Testing')
results = trainer.evaluate(eval_dataset=tokenized["validation"])
print(results)

# trainer.save_model('./anime_deberta_model')

# with open('label_encoder.pkl', 'wb') as f:
#     pickle.dump(label_encoder, f)