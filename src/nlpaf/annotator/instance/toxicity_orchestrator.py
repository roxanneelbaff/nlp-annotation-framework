
import pandas as pd

from nlpaf.annotator.orchestrator import PipeOrchestrator


class ToxicityOrchestrator(PipeOrchestrator):


    def save_annotations(self, annotated_texts: list) -> pd.DataFrame:
        out_arr = []
        for doc, context in annotated_texts:
            row = {self.input_id: context[self.input_id]}

            if doc._.toxicity is not None :
                for hedge_dict in doc._.toxicity:
                    row[f"toxicity_{hedge_dict['label']}_count"] = hedge_dict['count']
                    row[f"toxicity_{hedge_dict['label']}_ratio"] = hedge_dict['ratio']
                    row[f"toxicity_{hedge_dict['label']}"] = hedge_dict['score_mean']

                res_df = pd.DataFrame(doc._.toxicity  )

                row[f"toxicity_dominant"] =res_df.iloc[res_df["count"].argmax()]["label"]
                out_arr.append(row)
        out_df: pd.DataFrame = pd.DataFrame(out_arr)

        out_df.fillna(0, inplace=True)
        return out_df









