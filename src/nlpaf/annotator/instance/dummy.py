from nlpaf import logger
from nlpaf.annotator.orchestrator import PipeOrchestrator


class DummyDefaultPipeOrchestrator(PipeOrchestrator):
    pass


def customized_pipe_func(input):
    logger.debug("checking Dummy customized pipe")
    return input
