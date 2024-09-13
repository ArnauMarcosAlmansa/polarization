from abc import ABC, abstractmethod
from multiprocessing import Process
from typing import Any
import subprocess


class Task(ABC):
    @abstractmethod
    def run(self) -> None:
        pass


class CommandTask(Task):
    def __init__(self, command) -> None:
        super().__init__()
        self.command = command

    def run(self) -> None:
        subprocess.run(self.command, shell=True, check=True)


def command(command) -> Task:
    return CommandTask(command)


class FunctionTask(Task):
    def __init__(self, function) -> None:
        super().__init__()
        self.function = function

    def run(self) -> None:
        self.function()


def function(function) -> Task:
    return FunctionTask(function)


class ParallelTask(Task):
    def __init__(self, tasks: list[Task]) -> None:
        super().__init__()
        self.tasks = tasks

    def run(self) -> None:
        processes = []
        for task in self.tasks:
            p = Process(target=task.run)
            p.start()
            processes.append(p)

        for p in processes:
            p.join()


def parallel(tasks: list[Task]) -> Task:
    return ParallelTask(tasks)


class SequentialTask(Task):
    def __init__(self, tasks: list[Task]) -> None:
        super().__init__()
        self.tasks = tasks

    def run(self) -> None:
        for task in self.tasks:
            task.run()


def sequential(tasks: list[Task]) -> Task:
    return SequentialTask(tasks)


class Pipeline:
    def __init__(self, tasks: list[Task]):
        self.tasks = tasks

    def run(self) -> None:
        for task in self.tasks:
            task.run()
