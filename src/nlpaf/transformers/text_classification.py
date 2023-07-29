

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
import comet_ml
from comet_ml import Experiment
from sklearn.metrics import f1_score
import torch
from torch.optim import AdamW
import torch


@dataclasses.dataclass
class TextClassification:
    dataset: 'DatasetDict | str'
    id2label: Dict  # = dataclasses.field(default_factory=dict)
    label2id: Dict  # = dataclasses.field(default_factory=dict)
    training_key: str
    eval_key: str
    undersample: bool = False

    text_col: str = "text"

    pretrained_model_name: str = "microsoft/deberta-base"
    metric: str = "f1"
    averaged_metric: str = "macro"
    model_org: str = "notaphoenix"
    uncase: bool = False

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
    tokenizer_max_length: int = 1024
    tokenizer_padding: 'bool|str' = True  # max length per batch
    use_gpu: bool = True
    optimizer: str = "adamw_torch"
    tokenizer_special_tokens: dict = None
    is_fp16: bool = False
    gradient_accumulation_steps: int = 1,
    gradient_checkpointing: bool = False

    # EVALUATION #
    def reinit(self):
        print("Loading dataset")
        self.set_dataset()

        print("Setting Tokenizer and tokenizing the data...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.pretrained_model_name
            )
        if self.tokenizer_special_tokens is not None:
            self.tokenizer.add_special_tokens(self.tokenizer_special_tokens)
        print("Tokenizing Data")
        self.tokenize_data()

        self.set_data_collator()

        print("Setting evaluator...")
        self.set_evaluator()
        print("Setting model from pretrained-model...")
        self.set_model()

        print("Setting training args...")
        self.set_training_args()

        print("Setting trainer")
        self.init_trainer()

    def __post_init__(self):
        self.reinit()

    def set_dataset(self):
        if type(self.dataset) is str:
            self.dataset: DatasetDict = load_dataset(self.dataset)
        if self.undersample:
            print("undersampling")
            self._random_undersample()

    def _random_undersample(self, seed: int = 42):

        np.random.seed(seed)

        labels = np.array(self.dataset[self.training_key]['label'])

        print(np.unique(np.array(self.dataset[self.training_key]['label']),
                        return_counts=True))

        # Get the indices for each class
        label_indices: dict = {}
        min_length = -1
        for _label in np.unique(labels):
            label_indices[_label] = np.where(labels == _label)[0]
            length = len(label_indices[_label])
            if length < min_length or min_length == -1:
                min_length = length
                lowest_label = _label

        print("undersampling so all each class in the training set has length"
              f" {min_length}, same as class {lowest_label}")

        shuffled = {}
        for _label, _indices in label_indices.items():
            np.random.shuffle(_indices)
            shuffled[_label] = _indices[:min_length]

        balanced_indices = np.concatenate([v for _, v in shuffled.items()])
        self.dataset[self.training_key] = self.dataset[self.training_key].select(balanced_indices)

        print(np.unique(np.array(self.dataset[self.training_key]['label']),
                        return_counts=True))
        return self.dataset

    def set_evaluator(self):
        self.evaluator = evaluate.load(self.metric)

    def _get_example(self, index):
        return self.tokenized_data[self.eval_key][index]["text"]
    
    def compute_metrics(self, eval_pred):
        experiment = comet_ml.get_global_experiment() 
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        extra_params = {"average": self.averaged_metric} if \
            self.averaged_metric is not None else {}

        if experiment:
            epoch = int(experiment.curr_epoch) if experiment.curr_epoch is not None else 0
            experiment.set_epoch(epoch)
            experiment.log_confusion_matrix(
                y_true=labels,
                y_predicted=predictions,
                file_name=f"confusion-matrix-epoch-{epoch}.json",
                labels=[self.id2label[key] for key in sorted(self.id2label.keys())],
                index_to_example_function=self._get_example,
            )

        return self.evaluator.compute(predictions=predictions,
                                      references=labels,
                                      **extra_params)

    def preprocess_function(self, examples, tokenizer, text_col):
        #examples = examples[text_col].lower() if self.uncase else examples[text_col]
        #torch.autograd.profiler
        if self.uncase:
            examples["text"] = [text.lower() for text in examples["text"]]
        return tokenizer(examples,
                         truncation=True,
                         max_length=self.tokenizer_max_length,
                         padding=self.tokenizer_padding) # 'max_length', 'longest', True

    def tokenize_data(self):
        print(type(self.dataset))
        self.tokenized_data = self.dataset.map(
            self.preprocess_function,
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

        if self.use_gpu:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            #print_gpu_utilization()
            print(f"asked to move to GPU - Gpu exist? {torch.cuda.is_available()}")
            self.model = self.model.to(device)  # move model to GPU
            #print_gpu_utilization()

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
            metric_for_best_model=self.metric,
            push_to_hub=self.push_to_hub,
            hub_private_repo=self.hub_private_repo,
            report_to=self.report_to,
            do_eval=True,
            do_predict=True,
            do_train=True,
            run_name=self.output_dir,
            optim=self.optimizer,
            log_level="error",
            fp16=self.is_fp16,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            gradient_checkpointing=self.gradient_checkpointing
        )

    def init_trainer(self):
        self.trainer: Trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.tokenized_data[self.training_key],
            eval_dataset=self.tokenized_data[self.eval_key],
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
        )

    def train(self):
        result = self.trainer.train()
        if self.push_to_hub:
            self.trainer.push_to_hub()
        try:
            score = self.evaluate()
            print("score:", score)
        except Exception:
            print("Failed to push to hub")
        print_summary(result)
        return self.trainer

    # POST TRAINING
    def predict_all(self, text, task: str = "text-classification", use_current_model:bool=True):
        if use_current_model:
            result = self.trainer.predict(text) 
        else:
            classifier = pipeline(task, model=f"{self.model_org}/"
                                              f"{self.output_dir}")
            result = classifier(text)
        return result

    def predict_1label(self, text: str, use_current_model:bool=True):
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
        evaluation_results = None
        try:

            evaluation_results = self.trainer.evaluate(self.tokenized_data["test"])
            experiment = comet_ml.get_global_experiment() 
            experiment.log_metric("eval_best_model", evaluation_results)
        except Exception as e:
            print(f"error while evaluating on test set {e}")
    
        return evaluation_results

        # evaluate
    def model_init(self):
        return AutoModelForSequenceClassification.from_pretrained(
            self.pretrained_model_name,
            num_labels=len(self.id2label.keys())
            )
    
    @staticmethod
    def compute_objective(predictions, targets):
        # Convert predictions and targets to numpy arrays if they are not already
        predictions = predictions.numpy() if isinstance(predictions, torch.Tensor) else predictions
        targets = targets.numpy() if isinstance(targets, torch.Tensor) else targets
        
        # Calculate the macro F1 score
        macro_f1 = f1_score(targets, predictions, average='macro')
        
        return macro_f1
    
    def search_hyperparameters(self, n_trials: int = 10,
                               run_on_subset: bool = False,
                               num_shards: int = 10,
                               direction="maximize",
                               load_best_params: bool = True,
                               backend: str ="optuna",
                               hp_space_func=None):

        training_data = self.tokenized_data[self.training_key].shard(
            index=1,
            num_shards=num_shards) if run_on_subset \
            else self.tokenized_data[self.training_key]

        trainer = Trainer(
            model=None,
            model_init=self.model_init,
            args=self.training_args,
            train_dataset=training_data,
            eval_dataset=self.tokenized_data[self.eval_key],
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
        )

        best_run = trainer.hyperparameter_search(n_trials=n_trials,
                                                 direction=direction,
                                                 backend=backend,
                                                 hp_space=hp_space_func,)

        if load_best_params:
            
            for n, v in best_run.hyperparameters.items():
                print(f"Resetting {n}:{v}")
                setattr(self.trainer.args, n, v)
            # self.trainer.train()

        return best_run
