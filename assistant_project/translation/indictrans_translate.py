"""
Translation using IndicTrans2-200M models from AI4Bharat.
"""
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor
from language.detect_language import LanguageDetector

class IndicTranslator:
    """Translator for Indian languages using IndicTrans2."""
    
    def __init__(self):
        """Initialize translator with IndicTrans2 models."""
        self.detector = LanguageDetector()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading IndicTrans2 models on {self.device}...")
        try:
            self.indic_en_tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indictrans2-indic-en-dist-200M", trust_remote_code=True)
            self.indic_en_model = AutoModelForSeq2SeqLM.from_pretrained("ai4bharat/indictrans2-indic-en-dist-200M", trust_remote_code=True).to(self.device)
            self.en_indic_tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indictrans2-en-indic-dist-200M", trust_remote_code=True)
            self.en_indic_model = AutoModelForSeq2SeqLM.from_pretrained("ai4bharat/indictrans2-en-indic-dist-200M", trust_remote_code=True).to(self.device)
            self.ip = IndicProcessor(inference=True)
            print("✓ IndicTrans2 models loaded successfully")
            self.models_loaded = True
        except Exception as e:
            print(f"⚠ Error loading IndicTrans2 models: {e}")
            import traceback
            traceback.print_exc()
            self.models_loaded = False
    
    def translate(self, text, source_lang, target_lang):
        """Translate text between languages."""
        if source_lang == target_lang:
            return text
        if not self.models_loaded:
            print(f"Translation needed: {source_lang} -> {target_lang} (models not loaded)")
            return text
        try:
            print(f"Translating: {source_lang} -> {target_lang}")
            lang_map = {"en": "eng_Latn", "hi": "hin_Deva", "kn": "kan_Knda", "ta": "tam_Taml", "te": "tel_Telu"}
            src_lang = lang_map.get(source_lang, source_lang)
            tgt_lang = lang_map.get(target_lang, target_lang)
            if source_lang == "en":
                tokenizer = self.en_indic_tokenizer
                model = self.en_indic_model
            else:
                tokenizer = self.indic_en_tokenizer
                model = self.indic_en_model
            batch = self.ip.preprocess_batch([text], src_lang=src_lang, tgt_lang=tgt_lang)
            inputs = tokenizer(batch, truncation=True, padding="longest", return_tensors="pt", return_attention_mask=True).to(self.device)
            with torch.no_grad():
                generated_tokens = model.generate(**inputs, min_length=0, max_length=256, num_beams=5, num_return_sequences=1)
            generated_tokens = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            translations = self.ip.postprocess_batch(generated_tokens, lang=tgt_lang)
            translated_text = translations[0]
            print(f"✓ Translation complete")
            return translated_text
        except Exception as e:
            print(f"Translation error: {e}")
            import traceback
            traceback.print_exc()
            return text
    
    def auto_translate(self, text, target_lang):
        """Auto-detect source language and translate."""
        source_lang = self.detector.detect(text)
        return self.translate(text, source_lang, target_lang)
