
import pandas as pd

from textmining_utility.annotator.orchestrator import PipeOrchestrator



class EmotionPipeOrchestrator(PipeOrchestrator):
    _EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]


    def save_annotations(self, annotated_texts: list) -> pd.DataFrame:
        out_arr = []
        for doc, context in annotated_texts:
            row = {self.input_id: context[self.input_id]}

            if doc._.emotions is not None and len(doc._.emotions)>0:
                for emotion_score in doc._.emotions:
                    row[f"emotion_hartmann_{emotion_score['label']}"] = emotion_score['score_mean']
                    row[f"emotion_hartmann_{emotion_score['label']}_count"] = emotion_score['count']
                    row[f"emotion_hartmann_{emotion_score['label']}_ratio"] = emotion_score['ratio']

                row[f"emotion_hartmann_label"] = doc._.emotion_label
                out_arr.append(row)
        out_df: pd.DataFrame = pd.DataFrame(out_arr)
        out_df.fillna(0, inplace=True)
        return out_df









