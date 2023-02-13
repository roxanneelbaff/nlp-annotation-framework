import dataclasses
from abc import ABC, abstractmethod
from typing import ClassVar, Dict

import pandas as pd
import nlpaf.annotator.registered_pipes as registered
from nlpaf import logger, utils


class PipeOrchestrator(ABC):
    _ANNOTATION_TYPE: ClassVar = ["text", "url", "image"]
    _ANNOTATION_LEVEL: ClassVar = ["singleton", "collection"]

    def __init__(self, registered_pipes: Dict, orchestrator_config_id: str = None):
        self.name = (
            self.__class__.__name__
            if orchestrator_config_id is None
            else orchestrator_config_id
        )
        self.pipe_config = registered_pipes[self.name]

        self.annotation_level = self.pipe_config["level"]
        self.annotation_type = self.pipe_config["data_type"]
        self.annotated_column = self.pipe_config["annotated_column"]
        self.pipe_id_or_func = (
            self.pipe_config["pipe_id_or_func"]
            if "pipe_id_or_func" in self.pipe_config.keys()
            else None
        )
        self.pipe_path = (
            self.pipe_config["pipe_path"]
            if "pipe_path" in self.pipe_config.keys()
            else None
        )
        self.input_id = self.pipe_config["input_id"]

        self.load_pipe_component()

        is_valid, error_dic = self.validate()
        if is_valid:
            logger.info("orchestrator was initialized successfully")
        else:
            logger.error(
                f"orchestrator was initialized with the following errors {error_dic}"
            )

    def load_pipe_component(self):
        module = None
        if self.pipe_path is not None and len(self.pipe_path) > 0:
            logger.debug("loading {}".format(self.pipe_path))
            module = utils.load_module(self.pipe_path)
        return module

    def get_pipe_func(self):
        module = self.load_pipe_component()
        pipe_func = None
        if module is not None and self.pipe_id_or_func is not None:
            pipe_func = getattr(module, self.pipe_id_or_func)
        return pipe_func

    def validate(self):
        error_dict = {}
        if self.annotation_type not in PipeOrchestrator._ANNOTATION_TYPE:
            error_dict[
                "annotation_type"
            ] = "The annotation type should have one of the following values: {} but has the value {}".format(
                str(PipeOrchestrator._ANNOTATION_TYPE), self.annotation_type
            )
            logger.error(error_dict["annotation_type"])

        if self.annotation_level not in PipeOrchestrator._ANNOTATION_LEVEL:
            error_dict["annotation_level"] = (
                "The annotation level should have one of the following values: {} "
                "but has the value {}".format(
                    str(PipeOrchestrator._ANNOTATION_LEVEL), self.annotation_level
                )
            )
            logger.error(error_dict["annotation_level"])

        is_valid = not bool(error_dict)
        return is_valid, error_dict

    @abstractmethod
    def save_annotations(self, annotated_text: pd.DataFrame):
        pass
