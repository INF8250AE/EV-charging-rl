from abc import ABC, abstractmethod
import comet_ml
import os
from loguru import logger as console_logger


class Logger(ABC):
    @abstractmethod
    def get_job_id(self) -> str:
        pass

    @abstractmethod
    def set_name(self, name: str):
        pass

    @abstractmethod
    def add_tag(self, tag: str):
        pass

    @abstractmethod
    def log_metric(
        self, name: str, value: float, step: int = None, epoch: int = None, silent=False
    ):
        pass

    @abstractmethod
    def log_parameters(self, parameters: dict):
        pass

    @abstractmethod
    def log_video(self, file: str, step: int):
        pass

    @abstractmethod
    def log_image(self, image_data, name: str, step: int):
        pass

    @abstractmethod
    def log_code(self, folder: str):
        pass

    @abstractmethod
    def log_other(self, key: str, value):
        pass


class StdoutLogger(Logger):
    def get_job_id(self) -> str:
        return "undefined"

    def set_name(self, name: str):
        pass

    def add_tag(self, tag: str):
        pass

    def log_metric(
        self, name: str, value: float, step: int = None, epoch: int = None, silent=False
    ):
        if silent:
            return
        if step is None:
            number = epoch
        else:
            number = step
        console_logger.info(f"[{number}-{name}]: {value}")

    def log_parameters(self, parameters: dict):
        pass

    def log_video(self, file: str, step: int):
        pass

    def log_image(self, image_data, name: str, step: int):
        pass

    def log_code(self, folder: str):
        pass

    def log_other(self, key: str, value):
        console_logger.info(f"{key}: {value}")


class CometmlLogger(Logger):
    def __init__(self):
        api_key = os.getenv("COMET_API_KEY")
        project_name = os.getenv("COMET_PROJECT_NAME")
        workspace = os.getenv("COMET_WORKSPACE")

        disabled = api_key is None or project_name is None or workspace is None

        if disabled:
            console_logger.warning(
                "Warning : CometML Logging disabled, if you wish to use CometML logging please set the environment variables COMET_API_KEY, COMET_PROJECT_NAME and COMET_WORKSPACE"
            )

        offline_directory = os.getenv("COMET_OFFLINE_DIRECTORY", None)
        online = offline_directory is None or disabled

        self.online = online

        self.experiment = comet_ml.start(
            api_key=api_key,
            workspace=workspace,
            project_name=project_name,
            online=online,
            experiment_config=comet_ml.ExperimentConfig(
                disabled=disabled, offline_directory=offline_directory
            ),
        )

    def get_job_id(self) -> str:
        return self.experiment.get_key()

    def set_name(self, name: str):
        if name is not None:
            self.experiment.set_name(name)

    def add_tag(self, tag: str):
        self.experiment.add_tag(tag)

    def log_metric(
        self, name: str, value: float, step: int = None, epoch: int = None, silent=False
    ):
        self.experiment.log_metric(name=name, value=value, step=step, epoch=epoch)

    def log_parameters(self, parameters: dict):
        self.experiment.log_parameters(parameters)

    def log_video(self, file: str, step: int):
        self.experiment.log_video(file, step=step)

    def log_image(self, image_data, name: str, step: int):
        self.experiment.log_image(image_data=image_data, name=name, step=step)

    def log_code(self, folder: str):
        self.experiment.log_code(folder=folder)

    def log_other(self, key: str, value):
        self.experiment.log_other(key=key, value=value)
