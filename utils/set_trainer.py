import logging

import ignite.metrics as metrics
import matplotlib.pyplot as plt
import torch
from ignite.contrib.handlers import ProgressBar, FastaiLRFinder, LRScheduler
from ignite.contrib.handlers.param_scheduler import CosineAnnealingScheduler
from ignite.contrib.handlers.tensorboard_logger import *
from ignite.engine import Engine, Events, create_supervised_trainer
from ignite.handlers import EarlyStopping, TerminateOnNan, Timer
from torch.optim.lr_scheduler import MultiplicativeLR

from torch.cuda.amp import GradScaler, autocast

from utils import set_config
from utils.config import Config
from utils.ignite_metrics import GpuInfo_Fix as GpuInfo

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def score_fn(engine):
    score = engine.state.metrics["Accuracy"]
    return score


def activated_output_transform(output):
    y_pred, y = output
    y_pred = torch.sigmoid(y_pred)
    return y_pred, y


def dict_to_pair(data, **kwargs):
    inp, tar = data["input"], data["target"]
    return inp, tar


def _set_metrics():
    config = Config().get_instance()
    criterion = torch.nn.CrossEntropyLoss().to(device)
    val_metrics = {
        "Accuracy": metrics.Accuracy(device=device),
        "Precision": metrics.Precision(average=True,device=device),
        "Recall": metrics.Recall(average=True,device=device),        
        "F1-Score": metrics.Fbeta(beta=1.0,device=device),
        "Top-3_Error": metrics.TopKCategoricalAccuracy(k=3),
        "Top-5_Error": metrics.TopKCategoricalAccuracy(k=5),
    }
    return val_metrics, criterion


def _set_amp_trainer(net, dataset, optimizer, criterion):
    config = Config().get_instance()

    scaler = GradScaler()

    def train_step(engine, batch):
        x, y = batch["input"], batch["target"]

        optimizer.zero_grad()
        
        with autocast():
            y_pred = net(x)
            loss = criterion(y_pred, y)
            
        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()

        return loss.item()

    trainer = Engine(train_step)

    return trainer

def _get_evaluator(net, val_metrics):
    
    def _validation_step(engine, batch):
        net.eval()
        with torch.no_grad():
            x, y = batch["input"], batch["target"]
            with autocast():
                y_pred = net(x)
        
        net.train()
        return y_pred, y

    test_evaluator = Engine(_validation_step)
    
    for name,metric in val_metrics.items():
        metric.attach(test_evaluator,name)
    
    return test_evaluator

def _lr_schedule(epoch, stepsize = 30):
    if epoch % stepsize == 0:
        return 0.1
    else:
        return 1


def get_trainer(net, dataset, early_stop=False, scheduler=False, lrfinder=False):
    config = Config().get_instance()

    MixedPrecision = config["Trainer"]["MixedPrecision"]

    optimizer = set_config.choose_optimizer(net)

    val_metrics, criterion = _set_metrics()

    if MixedPrecision:
        trainer = _set_amp_trainer(net, dataset, optimizer, criterion)
    else:
        trainer = create_supervised_trainer(net, optimizer, criterion)
    
    test_evaluator = _get_evaluator(net, val_metrics)
    
    GpuInfo().attach(trainer, name='gpu')

    ProgressBar().attach(trainer, output_transform=lambda x: {"Loss": x})
    trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())

    timer = Timer(average=True)
    timer.attach(trainer, step=Events.EPOCH_COMPLETED(every=10))

    @trainer.on(Events.EPOCH_COMPLETED(every=10))
    def log_validation_results():
        test_evaluator.run(dataset.testloader)
        
    _m_scheduler = MultiplicativeLR(optimizer, _lr_schedule, last_epoch=-1, verbose=False)
    scheduler = LRScheduler(_m_scheduler)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, scheduler)

    return trainer, test_evaluator

