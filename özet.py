from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained("./saved_model")
tokenizer = AutoTokenizer.from_pretrained("./saved_model")

def summarize_text(text, max_length=300, min_length=150):
    inputs = tokenizer(text, return_tensors='pt', max_length=1024, truncation=True)
    summary_ids = model.generate(
        inputs['input_ids'],
        max_length=max_length,
        min_length=min_length,
        length_penalty=2.0, #Değer 2.0'dan büyükse daha uzun diziler, 2.0'dan küçükse daha kısa diziler tercih edilir.
        num_beams=4, #Beam search algoritmasında kullanılan beam sayısı. Beam search, en iyi sonuçları bulmak için kullandık.
        early_stopping=True #en iyi sonucu erken bulursak özetleme işlemini durdur.
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

example_text = input("Özetlemek istediğiniz metni giriniz: ")

summary = summarize_text(example_text)

print("\nSummary: \n", summary)
