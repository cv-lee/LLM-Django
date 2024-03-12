import nltk
from nltk.tokenize import sent_tokenize

from transformers import (
    T5ForConditionalGeneration, 
    T5Tokenizer,
)

class Translator:
    def __init__(self, model_name='google/madlad400-3b-mt'):        
        self.model = T5ForConditionalGeneration.from_pretrained(
            model_name, 
            device_map="auto"
        )
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        nltk.download('punkt')

    def _translate_sentence(self, text, target_lang):
        if target_lang == "EN":
            text = "<2en> " + text
        elif target_lang == "KO":
            text = "<2ko> " + text

        input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.model.device)
        outputs = self.model.generate(input_ids=input_ids, max_new_tokens=1000)
        res = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return res[0]
    
    def translate(self, texts, target_lang):
        sentences = sent_tokenize(texts)
        res = ""
        for text in sentences:
            res += self._translate_sentence(text, target_lang)
        return res

