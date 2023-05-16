

import dataclasses
from typing import Dict

from datasets import load_dataset
from datasets.dataset_dict import DatasetDict
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import evaluate
import numpy as np
from transformers import pipeline


@dataclasses.dataclass
class TextClassification:
    dataset: DatasetDict | str
    id2label: Dict  # = dataclasses.field(default_factory=dict)
    label2id: Dict  # = dataclasses.field(default_factory=dict)

    text_col: str = "text"

    pretrained_model_name: str = "microsoft/deberta-base"
    metric: str = "f1"
    averaged_metric: str = "macro"
    model_org: str = "notaphoenix"

    # Training specific
    output_dir: str = "my_awesome_model"
    learning_rate: float = 2e-5
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 16
    num_train_epochs: int = 2
    weight_decay: float = 0.01
    evaluation_strategy: str = "epoch"
    save_strategy: str = "epoch"
    load_best_model_at_end: bool = True
    push_to_hub: bool = True
    hub_private_repo: bool = True
    report_to: str = "all"  # comet_ml

    # EVALUATION #
    def reinit(self):
        print("Loading dataset")
        self.set_dataset()

        print("Setting Tokenizer and tokenizing the data...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.pretrained_model_name
            )

        print("Tokenizing Data")
        self.tokenize_data()

        self.set_data_collator()

        print("Setting evaluator...")
        self.set_evaluator()
        print("Setting model from pretrained-model...")
        self.set_model()

        print("Setting training args...")
        self.set_training_args()

    def __post_init__(self):
        self.reinit()

    def set_dataset(self):
        if type(self.dataset) is str:
            self.dataset: DatasetDict = load_dataset(self.dataset)

    def set_evaluator(self):
        self.evaluator = evaluate.load(self.metric)

    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        extra_params = {"average": self.averaged_metric} if \
            self.averaged_metric is not None else {}

        return self.evaluator.compute(predictions=predictions,
                                      references=labels,
                                      **extra_params)

    def preprocess_function(examples, tokenizer, text_col):
        return tokenizer(examples[text_col],
                         truncation=True)

    def tokenize_data(self):
        print(type(self.dataset))
        self.tokenized_data = self.dataset.map(
            TextClassification.preprocess_function,
            fn_kwargs={
                "tokenizer": self.tokenizer,
                "text_col": self.text_col
                },
            batched=True)
        return self.tokenized_data

    def set_data_collator(self):
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

    def set_model(self):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.pretrained_model_name,
            num_labels=len(self.id2label.keys()),
            id2label=self.id2label,
            label2id=self.label2id
        )

    def set_training_args(self):
        self.training_args = TrainingArguments(
            output_dir=self.output_dir,
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.per_device_train_batch_size,
            per_device_eval_batch_size=self.per_device_eval_batch_size,
            num_train_epochs=self.num_train_epochs,
            weight_decay=self.weight_decay,
            evaluation_strategy=self.evaluation_strategy,
            save_strategy=self.save_strategy,
            load_best_model_at_end=self.load_best_model_at_end,
            push_to_hub=self.push_to_hub,
            hub_private_repo=self.hub_private_repo,
            report_to=self.report_to,
            do_eval=True,
            do_predict=True
        )

    def train(self):

        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.tokenized_data["train"],
            eval_dataset=self.tokenized_data["test"],
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
        )

        self.trainer.train()
        try:
            self.trainer.push_to_hub()
        except Exception:
            print("Failed to push to hub")

        return self.trainer

    # POST TRAINING
    def predict_all(self, text, task: str = "text-classification"):
        classifier = pipeline(task, model=f"{self.model_org}/"
                                          f"{self.output_dir}")
        return classifier(text)

    def predict_1label(self, text: str):
   
        tokenizer = AutoTokenizer.from_pretrained(f"{self.model_org}"
                                                  f"/{self.output_dir}")
        inputs = tokenizer(text, return_tensors="pt")

        model = AutoModelForSequenceClassification.from_pretrained(f"{self.model_org}/{self.output_dir}")
        with torch.no_grad():
            logits = model(**inputs).logits
        predicted_class_id = logits.argmax().item()
        label = model.config.id2label[predicted_class_id]
        return label
    
    def evaluate(self):
        # Loop over test set 
        predictions = []
        labels = []
        for x in range(1,100):
            text = ""
            predicted = self.predict_1label(text)
            predictions.append(predicted)

        score: float = round(self.evaluator.compute(
                                    predictions=predictions,
                                    references=labels,
                                    average=self.averaged_metric), 2)
        return score

        # evaluate

    def search_hyperparameters(self, n_trials: int = 10,
                               run_on_subset: bool = False,
                               num_shards: int = 10,
                               direction="maximize",
                               load_best_params: bool = True):
        def model_init(self):
            return AutoModelForSequenceClassification.from_pretrained(
                self.pretrained_model_name,
                num_labels=len(self.id2label.keys())
                )
        
        training_data = self.tokenized_data["train"].shard(index=1, 
                                                           num_shards=num_shards) if run_on_subset \
                                                           else self.tokenized_data["train"]

        trainer = Trainer(
            model_init=model_init,
            args=self.training_args,
            train_dataset=training_data,
            eval_dataset=self.tokenized_data["test"],
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics
        )

        best_run = trainer.hyperparameter_search(n_trials=10,
                                                 direction=direction)
               
        if load_best_params:
            for n, v in best_run.hyperparameters.items():
                print(f"Resetting {n}:{v}")
                setattr(self.trainer.args, n, v)
            self.trainer.train()

        return best_run
