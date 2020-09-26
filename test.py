from utils import load_data, set_config
import numpy as np
import torch
import torchvision
import torch.nn as nn
from collections import namedtuple, defaultdict
from functools import partial
import copy
import ignite
from ignite.engine import Events, Engine, create_supervised_trainer, create_supervised_evaluator
from torch.cuda.amp import GradScaler, autocast
import time
from utils.config import Config
from ignite.contrib.handlers import ProgressBar
from ignite.metrics import Accuracy, Loss
from ignite.handlers import Timer

config = Config().get_instance()

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

cpu = torch.device("cpu")


def cov(X):
    X = X / np.sqrt(X.size(0) - 1)
    return X.t() @ X


def patches(data, patch_size=(3, 3), dtype=torch.float32):
    h, w = patch_size
    c = data.size(1)
    return data.unfold(2, h, 1).unfold(3, w, 1).transpose(1, 3).reshape(-1, c, h, w).to(dtype)


def eigens(patches):
    n, c, h, w = patches.shape
    Σ = cov(patches.reshape(n, c * h * w))
    Λ, V = torch.symeig(Σ, eigenvectors=True)
    return Λ.flip(0), V.t().reshape(c * h * w, c, h, w).flip(0)


"""class Timer:
    def __init__(self, synch=None):
        self.synch = synch or (lambda: None)
        self.synch()
        self.times = [time.perf_counter()]
        self.total_time = 0.0

    def __call__(self, update_total=True):
        self.synch()
        self.times.append(time.perf_counter())
        delta_t = self.times[-1] - self.times[-2]
        if update_total:
            self.total_time += delta_t
        return delta_t"""


dataset = load_data.get_data()  # downloads dataset

# Λ, V = eigens(patches(train_set["data"][:10000, :, 4:-4, 4:-4]))  # center crop to remove padding

# torch.save(Λ, "./data/datasets/cifar10/eigens1.pt")
# torch.save(V, "./data/datasets/cifar10/eigens2.pt")

"""t = Timer(synch=torch.cuda.synchronize)
batches = dataset.trainloader
print()
for epoch in range(5):
    for idx, batch in enumerate(batches):
        print(batch)
        input, target = batch
print(f"{t():.3f}s")"""


def dict_to_pair(data, **kwargs):
    inp, tar = data["input"], data["target"]
    return inp, tar


net = set_config.choose_architecture().to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = set_config.choose_optimizer(net)

scaler = GradScaler()


def train_step(engine, batch):
    x, y = batch["input"], batch["target"]

    optimizer.zero_grad()

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


metrics = {"Accuracy": Accuracy(), "Loss": Loss(criterion)}

trainer = Engine(train_step)
evaluator = create_supervised_evaluator(
    net, metrics=metrics, device=device, non_blocking=True, prepare_batch=dict_to_pair
)
ProgressBar(persist=True).attach(trainer, output_transform=lambda out: {"batch loss": out})

timer = Timer(average=True)
timer.attach(trainer, step=Events.EPOCH_COMPLETED)


def log_metrics(engine, title):
    for name in metrics:
        print("\t{} {}: {:.2f}".format(title, name, engine.state.metrics[name]))


@trainer.on(Events.EPOCH_COMPLETED)
def run_validation(_):
    print("- Mean elapsed time for 1 epoch: {}".format(timer.value()))
    print("- Metrics:")
    with evaluator.add_event_handler(Events.COMPLETED, log_metrics, "Train"):
        evaluator.run(dataset.trainloader)

    with evaluator.add_event_handler(Events.COMPLETED, log_metrics, "Test"):
        evaluator.run(dataset.testloader)


trainer.run(dataset.trainloader, max_epochs=config["Setup"]["Epochs"])
