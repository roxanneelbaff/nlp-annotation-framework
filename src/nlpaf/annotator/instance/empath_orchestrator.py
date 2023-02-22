import pandas as pd

from nlpaf.annotator.orchestrator import PipeOrchestrator


class EmpathPipeOrchestrator(PipeOrchestrator):
    def save_annotations(self, annotated_texts: list) -> pd.DataFrame:
        out_arr = []
        for doc, context in annotated_texts:
            row = {self.input_id: context[self.input_id]}

            for category, count in doc._.empath_count.items():
                row[f"empath_{category}_count"] = doc._.empath_count[category]
                row[f"empath_{category}_ratio"] = doc._.empath_ratio[category]
                row[f"empath_{category}_position"] = doc._.empath_positions[
                    category
                ]
                row[f"empath_{category}_word"] = doc._.empath_words[category]
            out_arr.append(row)
        out_df: pd.DataFrame = pd.DataFrame(out_arr)

        return out_df
