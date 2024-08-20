import os
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq

data_path = r"C:\Users\karad\Desktop\cnn_dailymail"
# https://www.kaggle.com/datasets/gowrishankarp/newspaper-text-summarization-cnn-dailymail adresinden veri setini indriebilirsiniz.

train_data = pd.read_csv(os.path.join(data_path, "train.csv")).sample(1000, random_state=42)
val_data = pd.read_csv(os.path.join(data_path, "validation.csv")).sample(1000, random_state=42)
test_data = pd.read_csv(os.path.join(data_path, "test.csv")).sample(1000, random_state=42)

train_dataset = Dataset.from_pandas(train_data)
val_dataset = Dataset.from_pandas(val_data)
test_dataset = Dataset.from_pandas(test_data)

dataset = DatasetDict({
    'train': train_dataset,
    'validation': val_dataset,
    'test': test_dataset
})

model_name = "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def preprocess_function(examples):
    inputs = examples['article']
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples['highlights'], max_length=128, truncation=True)

    model_inputs['labels'] = labels['input_ids']
    return model_inputs

tokenized_train_dataset = dataset['train'].map(preprocess_function, batched=True)
tokenized_val_dataset = dataset['validation'].map(preprocess_function, batched=True)
tokenized_test_dataset = dataset['test'].map(preprocess_function, batched=True)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch", #Modelin ne zaman değerlendirileceğini belirler. Her epoch'ta değerlendirme yapacak.
    learning_rate=2e-5, #öğrenme oranı
    per_device_train_batch_size=2, #eğitim için kullanılacak veri yığınlarının boyutu
    per_device_eval_batch_size=2, #değerlendirme için kullanılacak veri yığınlarının boyutu
    weight_decay=0.01, #aşırı uyum (overfitting) sorununu azaltmak ve daha iyi genelleştirilmiş bir model elde etmek için kullandık.
    save_total_limit=3, #Eğitim sırasında kaydedilecek kontrol noktalarının maksimum sayısı
    num_train_epochs=3,
    predict_with_generate=True, #Modelin tahmin yaparken üretim kullanıp kullanmayacağını belirtir.
    fp16=False,  # fp16 karışık hassasiyet eğitimini devre dışı bıraktım. Bilgisayarımın özellikleri ile uyumlu çalışmıyor.
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args, #eğitim sürecinin çeşitli yönlerini (öğrenme oranı, batch boyutu, epoch sayısı, ağırlık azalması vb.) yapılandırmak için
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator, #Modelin eğitim ve değerlendirme sürecinde veri örneklerini uygun şekilde düzenlenmesi
)

trainer.train()

model.save_pretrained("./saved_model")
tokenizer.save_pretrained("./saved_model")
