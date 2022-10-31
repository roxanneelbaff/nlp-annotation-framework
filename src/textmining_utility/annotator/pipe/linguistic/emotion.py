from simpletransformers.ner import NERModel
from spacy.language import Language
from spacy.tokens import Token, Span, Doc
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

# REQUIRES sentencizer

@Language.factory("emotion_hartmann_component")
class EmotionHartmannFactory:
    _EMOTION_DOC_KEY: str = "emotion_hartmann_{}"
    _EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
    def __init__(self, nlp: Language, name: str):
        self.nlp = nlp

        self.transformer_nlp = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            return_all_scores=True
        )


        for emotion in _EMOTIONS:
            if not Doc.has_extension(_EMOTION_DOC_KEY.format(emotion)):
                Doc.set_extension(_EMOTION_DOC_KEY.format(emotion), default=None)

    def __call__(self, doc):

        raw_results = self.transformer_nlp(doc.text)

         # Dominant Label
        results_df = pd.DataFrame(raw_results)
        dominant_emotion = results_df.iloc[results_df["score"].argmax()]["label"]
        doc._.set(_EMOTION_DOC_KEY.format("dominant"), dominant_emotion)


        doc._.set(_EMOTION_DOC_KEY.format("scores"), raw_results) # [{"label": emotion, "score": 0.xyz]
        return doc