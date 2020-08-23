import logging
import time
from datetime import datetime

import matplotlib.pyplot as plt

from ignite.handlers import EarlyStopping, TerminateOnNan
import ignite.metrics as metrics
import torch
import torch.nn as nn
from ignite.contrib.handlers import ProgressBar, FastaiLRFinder
from ignite.contrib.handlers.param_scheduler import CosineAnnealingScheduler
from ignite.contrib.handlers.tensorboard_logger import *
from ignite.engine import Events
from ignite.engine import create_supervised_evaluator
from ignite.engine import create_supervised_trainer
from torch.utils.tensorboard import SummaryWriter

from utils import set_config
from utils import visualize
from utils.ignite_metrics import ROC_AUC
from utils.config import Config

config = Config.get_instance()

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

criterion = nn.CrossEntropyLoss()

val_metrics = {
    "Accuracy": metrics.Accuracy(),
    "Running_Average_Accuracy": metrics.RunningAverage(metrics.Accuracy()),
    "Loss": metrics.Loss(criterion),
    "Running_Average_Loss": metrics.RunningAverage(metrics.Loss(criterion)),
    "Precision": metrics.Precision(average=True),
    "Recall": metrics.Recall(average=True),
    "F1-Score": metrics.Fbeta(beta=1.0),
    "ROC_AUC": ROC_AUC(output_transform=activated_output_transform),
}


def score_fn(engine):
    score = engine.state.metrics["Accuracy"]
    return score


def activated_output_transform(output):
    y_pred, y = output
    soft = nn.Softmax(dim=1)
    y_pred = soft(y_pred)
    return y_pred, y


def get_trainer(net, dataset, early_stop=False, scheduler=False, lrfinder=False):

    optimizer = set_config.choose_optimizer(net)

    trainer = create_supervised_trainer(net, optimizer, criterion, device=device)
    train_evaluator = create_supervised_evaluator(net, metrics=val_metrics, device=device)
    test_evaluator = create_supervised_evaluator(net, metrics=val_metrics, device=device)

    if lrfinder:
        find_lr(
            trainer,
            dataset,
            nruns=config["Trainer"]["LRFinderCycles"],
            plot=config["Trainer"]["LRFinderPlot"],
        )
        exit()

    if early_stop:
        stopper = EarlyStopping(patience=15, score_function=score_fn, trainer=test_evaluator)
        test_evaluator.add_event_handler(Events.COMPLETED, stopper)

    if scheduler:
        scheduler = CosineAnnealingScheduler(
            optimizer, "lr", 1e-1, 1e-3, len(dataset.trainloader), start_value=10000
        )
        trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

    ProgressBar().attach(trainer, output_transform=lambda x: {"Loss": x})
    trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        train_evaluator.run(dataset.trainloader)
        metrics = train_evaluator.state.metrics
        logging.info(
            "Training Results - Epoch:\t{}\nAverage Accuracy:\t{:.2f}\nRunning Average Accuracy:\t{:.2f}\nAverage Loss:\t{:.2f}\nRunning Average Loss:\t{:.2f}\nAverage Precision:\t{:.2f}\nAverage Recall:\t{:.2f}\nAverage F1-Score:\t{:.2f}\nAverage ROC AUC:\t{:2f}".format(
                trainer.state.epoch,
                metrics["Accuracy"],
                metrics["Running_Average_Accuracy"],
                metrics["Loss"],
                metrics["Running_Average_Loss"],
                metrics["Precision"],
                metrics["Recall"],
                metrics["F1-Score"],
                metrics["ROC_AUC"],
            )
        )

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        test_evaluator.run(dataset.testloader)
        metrics = test_evaluator.state.metrics
        logging.info(
            "Validation Results - Epoch:\t{}\nAverage Accuracy:\t{:.2f}\nRunning Average Accuracy:\t{:.2f}\nAverage Loss:\t{:.2f}\nRunning Average Loss:\t{:.2f}\nAverage Precision:\t{:.2f}\nAverage Recall:\t{:.2f}\nAverage F1-Score:\t{:.2f}\nAverage ROC AUC:\t{:2f}".format(
                trainer.state.epoch,
                metrics["Accuracy"],
                metrics["Running_Average_Accuracy"],
                metrics["Loss"],
                metrics["Running_Average_Loss"],
                metrics["Precision"],
                metrics["Recall"],
                metrics["F1-Score"],
                metrics["ROC_AUC"],
            )
        )

    return trainer, train_evaluator, test_evaluator


def find_lr(trainer, dataset, nruns=1, plot=False):

    lr_finder = FastaiLRFinder()

    upper_bound = 0
    lower_bound = 0
    lr_bounds = []
    for i in range(nruns):
        with lr_finder.attach(trainer) as trainer_with_lr_finder:
            trainer_with_lr_finder.run(dataset.trainloader)

        if plot:
            lr_finder.plot()
            plt.show()

        if nruns > 1:
            lr_bounds.append(lr_finder.lr_suggestion())
        else:
            logging.info("Suggested LearnRate:\t{:.2f}".format(lr_finder.lr_suggestion()))
            return

    upper_bound = max(lr_bounds)
    lower_bound = min(lr_bounds)

    logging.info(
        "Suggested LearnRate Boundaries:\t{:.2f} - {:.2f}".format(lower_bound, upper_bound)
    )

    return
