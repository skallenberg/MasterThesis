import logging
import time
from datetime import datetime

import matplotlib.pyplot as plt

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


def activated_output_transform(output):
    y_pred, y = output
    soft = nn.Softmax(dim=1)
    y_pred = soft(y_pred)
    return y_pred, y


def train(net, dataset):

    if torch.cuda.device_count() > 1:
        writer_name = (
            net.module.name
            + "_"
            + dataset.name
            + "_"
            + datetime.now().strftime("%m-%d-%Y-%H:%M:%S")
        )
        net.module.writer = writer_name
    else:
        writer_name = (
            net.name + "_" + dataset.name + "_" + datetime.now().strftime("%m-%d-%Y-%H:%M:%S")
        )
        net.writer = writer_name

    tb_logger = TensorboardLogger(log_dir="./data/models/logs/runs/" + writer_name)

    optimizer = set_config.choose_optimizer(net)
    criterion = nn.CrossEntropyLoss()

    trainer = create_supervised_trainer(net, optimizer, criterion, device=device)
    ProgressBar(persist=True).attach(trainer, metric_names="all")

    roc_auc = ROC_AUC(output_transform=activated_output_transform)
    val_metrics = {
        "Accuracy": metrics.Accuracy(),
        "Loss": metrics.Loss(criterion),
        "Precision": metrics.Precision(average=True),
        "Recall": metrics.Recall(average=True),
        "F1-Score": metrics.Fbeta(beta=1.0),
        "ROC_AUC": roc_auc,
    }

    train_evaluator = create_supervised_evaluator(net, metrics=val_metrics, device=device)
    test_evaluator = create_supervised_evaluator(net, metrics=val_metrics, device=device)

    scheduler = CosineAnnealingScheduler(optimizer, "lr", 0.3, 0.01, len(dataset.trainloader))
    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

    @trainer.on(Events.ITERATION_COMPLETED(every=config["Trainer"]["log_interval"]))
    def log_training_metrics(engine):
        logging.info("Epoch[{}] Loss: {:.2f}".format(engine.state.epoch, engine.state.output))

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        train_evaluator.run(dataset.trainloader)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        test_evaluator.run(dataset.testloader)

    tb_logger.attach_output_handler(
        trainer, event_name=Events.ITERATION_COMPLETED, tag="training", metric_names="all",
    )

    tb_logger.attach_output_handler(
        train_evaluator,
        event_name=Events.EPOCH_COMPLETED,
        tag="train_validation",
        metric_names="all",
        global_step_transform=global_step_from_engine(trainer),
    )

    tb_logger.attach_output_handler(
        test_evaluator,
        event_name=Events.EPOCH_COMPLETED,
        tag="validation",
        metric_names="all",
        global_step_transform=global_step_from_engine(trainer),
    )

    tb_logger.attach(
        trainer, event_name=Events.ITERATION_COMPLETED, log_handler=WeightsScalarHandler(net),
    )

    tb_logger.attach(
        trainer, event_name=Events.EPOCH_COMPLETED, log_handler=WeightsHistHandler(net),
    )

    tb_logger.attach(
        trainer, event_name=Events.ITERATION_COMPLETED, log_handler=GradsScalarHandler(net),
    )

    tb_logger.attach(trainer, event_name=Events.EPOCH_COMPLETED, log_handler=GradsHistHandler(net))

    trainer.run(dataset.trainloader, max_epochs=5)
