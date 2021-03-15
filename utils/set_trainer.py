import logging
import time
from datetime import datetime

import ignite.metrics as metrics
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from ignite.contrib.handlers import FastaiLRFinder
from ignite.contrib.handlers import ProgressBar
from ignite.contrib.handlers.param_scheduler import CosineAnnealingScheduler
from ignite.contrib.handlers.tensorboard_logger import *
from ignite.engine import Engine
from ignite.engine import Events
from ignite.engine import create_supervised_evaluator
from ignite.engine import create_supervised_trainer
from ignite.handlers import EarlyStopping
from ignite.handlers import TerminateOnNan
from ignite.handlers import Timer
import os
import pandas as pd
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast

from utils import set_config
from utils.config import Config
from utils.cross_entropy import CrossEntropyLoss
from utils.ignite_metrics import ROC_AUC
from utils.mixup import *

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def score_fn(engine):
    score = engine.state.metrics["Accuracy"]
    return score


def activated_output_transform(output):
    y_pred, y = output
    soft = nn.Softmax(dim=1)
    y_pred = soft(y_pred)
    return y_pred, y


def dict_to_pair(data, **kwargs):
    inp, tar = data["input"], data["target"]
    return inp, tar


def _set_metrics():
    config = Config().get_instance()
    if config["Trainer"]["LabelSmoothing"]:
        criterion = CrossEntropyLoss(smooth_eps=0.2).to(device)
    else:
        criterion = torch.nn.CrossEntropyLoss().to(device)
    val_metrics = {
        "Accuracy": metrics.Accuracy(),
        "Running_Average_Accuracy": metrics.RunningAverage(metrics.Accuracy()),
        "Loss": metrics.Loss(criterion),
        "Running_Average_Loss": metrics.RunningAverage(metrics.Loss(criterion)),
        "Precision": metrics.Precision(average=True),
        "Recall": metrics.Recall(average=True),
    }
    return val_metrics, criterion


def _set_amp_trainer(net, dataset, optimizer, criterion):
    config = Config().get_instance()

    MixUpData = config["Trainer"]["MixUp"]
    scaler = GradScaler()

    def train_step(engine, batch):
        x, y = batch["input"], batch["target"]

        optimizer.zero_grad()

        if MixUpData:
            x, y_a, y_b, lam = mixup_data(x, y, 0.4)
            x, y_a, y_b = map(torch.autograd.Variable, (x, y_a, y_b))
            with autocast():
                y_pred = net(x)
                loss = mixup_criterion(criterion, y_pred, y_a, y_b, lam)
        else:
            # Runs the forward pass with autocasting.
            with autocast():
                y_pred = net(x)
                loss = criterion(y_pred, y)
        # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
        # Backward passes under autocast are not recommended.
        # Backward ops run in the same precision that autocast used for corresponding forward ops.
        scaler.scale(loss).backward()

        # scaler.step() first unscales the gradients of the optimizer's assigned params.
        # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
        # otherwise, optimizer.step() is skipped.
        scaler.step(optimizer)

        # Updates the scale for next iteration.
        scaler.update()

        return loss.item()

    trainer = Engine(train_step)

    return trainer


def _set_mixup_trainer(net, dataset, optimizer, criterion):
    config = Config().get_instance()

    def train_step(engine, batch):
        x, y = batch["input"], batch["target"]

        optimizer.zero_grad()

        x, y_a, y_b, lam = mixup_data(x, y, 0.4)
        x, y_a, y_b = map(torch.autograd.Variable, (x, y_a, y_b))
        y_pred = net(x)
        loss = mixup_criterion(criterion, y_pred, y_a, y_b, lam)

        loss.backward()

        optimizer.step()

        return loss.item()

    trainer = Engine(train_step)

    return trainer


def get_trainer(net, dataset, early_stop=False, scheduler=False, lrfinder=False):
    config = Config().get_instance()

    MixedPrecision = config["Trainer"]["MixedPrecision"]
    MixUp = config["Trainer"]["MixUp"]

    optimizer = set_config.choose_optimizer(net)

    val_metrics, criterion = _set_metrics()

    if MixedPrecision:
        trainer = _set_amp_trainer(net, dataset, optimizer, criterion)
    elif MixUp:
        trainer = _set_mixup_trainer(net, dataset, optimizer, criterion)
    else:
        trainer = create_supervised_trainer(net, optimizer, criterion)

    train_evaluator = create_supervised_evaluator(
        net, metrics=val_metrics,device=device, prepare_batch=dict_to_pair, non_blocking=True
    )
    test_evaluator = create_supervised_evaluator(
        net, metrics=val_metrics,device=device, prepare_batch=dict_to_pair, non_blocking=True
    )

    if lrfinder:
        find_lr(
            net,
            optimizer,
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
        scheduler = CosineAnnealingScheduler(optimizer, "lr", 0.01, 0.0005, cycle_size=len(dataset.trainloader))
        trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

    ProgressBar().attach(trainer, output_transform=lambda x: {"Loss": x})
    trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())

    timer = Timer(average=True)
    timer.attach(trainer, step=Events.EPOCH_COMPLETED)
    
    # tr_frame = pd.DataFrame(columns=["Epoch","Time","Accuracy", "Running_Average_Accuracy", "Loss","Running_Average_Loss","Precision","Recall","F1-Score","ROC_AUC"])
    # os.mkdir("./data/models/logs/csv/" + net.writer+"/")
    # tr_frame.to_csv("./data/models/logs/csv/" + net.writer +"/TR.csv",index=False)
    # val_frame = pd.DataFrame(columns=["Epoch","Time","Accuracy", "Running_Average_Accuracy", "Loss","Running_Average_Loss","Precision","Recall","F1-Score","ROC_AUC"])
    # val_frame.to_csv("./data/models/logs/csv/"+ net.writer+"/VAL.csv",index=False)
    
    # @trainer.on(Events.EPOCH_COMPLETED(every=2))
    # def log_training_results(engine):
    #     train_evaluator.run(dataset.trainloader)
    #     metrics = train_evaluator.state.metrics
    #     #tr_frame = pd.read_csv("./data/models/logs/csv/"+ net.writer +"/TR.csv", header=0)
    #     #tr_frame = tr_frame.append({"Epoch":trainer.state.epoch,"Time":timer.value(),"Accuracy":metrics["Accuracy"], "Running_Average_Accuracy":metrics["Running_Average_Accuracy"], "Loss":metrics["Loss"],"Running_Average_Loss":metrics["Running_Average_Loss"],"Precision":metrics["Precision"],"Recall":metrics["Recall"],"F1-Score":metrics["F1-Score"],"ROC_AUC":metrics["ROC_AUC"]}, ignore_index=True)
    #     #tr_frame.to_csv("./data/models/logs/csv/"+ net.writer+"/TR.csv",index=False)
    #     # logging.info("- Mean elapsed time for 1 trainings epoch: {}".format(timer.value()))
    #     # logging.info(
    #     #     "Training Results - Epoch:\t{}\nAverage Accuracy:\t{:.2f}\nRunning Average Accuracy:\t{:.2f}\nAverage Loss:\t{:.2f}\nRunning Average Loss:\t{:.2f}\nAverage Precision:\t{:.2f}\nAverage Recall:\t{:.2f}\nAverage F1-Score:\t{:.2f}\nAverage ROC AUC:\t{:2f}".format(
    #     #         trainer.state.epoch,
    #     #         metrics["Accuracy"],
    #     #         metrics["Running_Average_Accuracy"],
    #     #         metrics["Loss"],
    #     #         metrics["Running_Average_Loss"],
    #     #         metrics["Precision"],
    #     #         metrics["Recall"],
    #     #         metrics["F1-Score"],
    #     #         metrics["ROC_AUC"],
    #     #     )
    #     # )

    # @trainer.on(Events.EPOCH_COMPLETED(every=2))
    # def log_validation_results(engine):
    #     test_evaluator.run(dataset.testloader)
    #     metrics = test_evaluator.state.metrics
    #     # val_frame = pd.read_csv("./data/models/logs/csv/"+ net.writer +"/VAL.csv", header=0)
    #     # val_frame = val_frame.append({"Epoch":trainer.state.epoch,"Time":timer.value(),"Accuracy":metrics["Accuracy"], "Running_Average_Accuracy":metrics["Running_Average_Accuracy"], "Loss":metrics["Loss"],"Running_Average_Loss":metrics["Running_Average_Loss"],"Precision":metrics["Precision"],"Recall":metrics["Recall"],"F1-Score":metrics["F1-Score"],"ROC_AUC":metrics["ROC_AUC"]}, ignore_index=True)
    #     # val_frame.to_csv("./data/models/logs/csv/"+ net.writer+"/VAL.csv",index=False)
    #     logging.info("- Mean elapsed time for 1 validation epoch: {}".format(timer.value()))
    #     logging.info(
    #         "Validation Results - Epoch:\t{}\nAverage Accuracy:\t{:.2f}\nRunning Average Accuracy:\t{:.2f}\nAverage Loss:\t{:.2f}\nRunning Average Loss:\t{:.2f}\nAverage Precision:\t{:.2f}\nAverage Recall:\t{:.2f}".format(
    #             trainer.state.epoch,
    #             metrics["Accuracy"],
    #             metrics["Running_Average_Accuracy"],
    #             metrics["Loss"],
    #             metrics["Running_Average_Loss"],
    #             metrics["Precision"],
    #             metrics["Recall"],
    #         )
    #     )

    return trainer, train_evaluator, test_evaluator


def find_lr(model, optimizer, trainer, dataset, nruns=1, plot=False):

    lr_finder = FastaiLRFinder()

    upper_bound = 0
    lower_bound = 0
    lr_bounds = []
    to_save = {"model": model, "optimizer": optimizer}
    for i in range(nruns):
        with lr_finder.attach(trainer, to_save=to_save) as trainer_with_lr_finder:
            trainer_with_lr_finder.run(dataset.trainloader)

        if plot:
            lr_finder.plot()
            plt.show()

        if nruns > 1:
            lr_bounds.append(lr_finder.lr_suggestion())
            logging.info("Suggested LearnRate:\t{:.2f}".format(lr_finder.lr_suggestion()))
        else:
            logging.info("Suggested LearnRate:\t{:.2f}".format(lr_finder.lr_suggestion()))
            return

    upper_bound = max(lr_bounds)
    lower_bound = min(lr_bounds)

    logging.info(
        "Suggested LearnRate Boundaries:\t{:.2f} - {:.2f}".format(lower_bound, upper_bound)
    )

    return
