import logging
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from torch.utils.tensorboard import SummaryWriter

from utils import set_config
from utils import visualize
from utils.config import Config

config = Config.get_instance()
criterion = nn.CrossEntropyLoss()

if torch.cuda.is_available():
    device = torch.device(config["Setup"]["Device"])
    if torch.cuda.device_count() > 1:
        config["Setup"]["Parallel"] = 1
    else:
        config["Setup"]["Parallel"] = 0
else:
    device = torch.device("cpu")


def train(net, dataset, return_data=False):

    if config["Setup"]["Parallel"] == 1:
        writer_name = (
            net.module.name
            + "_"
            + dataset.name
            + "_"
            + datetime.now().strftime("%m-%d-%Y-%H:%M:%S")
        )
        writer = SummaryWriter("./data/models/log/runs/" + writer_name + "/train")

        net.module.writer = writer_name
    else:
        writer_name = (
            net.name + "_" + dataset.name + "_" + datetime.now().strftime("%m-%d-%Y-%H:%M:%S")
        )
        writer = SummaryWriter("./data/models/log/runs/" + writer_name + "/train")

        net.writer = writer_name

    optimizer = set_config.choose_optimizer(net)

    loss_list = []
    accuracy_list = []
    f_score_list = []
    iter_list = []

    _global_start = time.time()

    logging.info("Started Training")

    _epoch_process_time = 0

    avg_loss_window = []

    for epoch in range(config["Setup"]["Epochs"]):

        running_loss = 0.0
        correct = 0
        total = 0
        cumulated_labels = []
        cumulated_predictions = []

        _epoch_start = time.time()

        for i, data in enumerate(dataset.trainloader):
            iteration = epoch * len(list(dataset.trainloader)) + i
            optimizer.zero_grad()

            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = net(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            curr_loss = loss.item()

            del loss

            running_loss += curr_loss
            avg_loss_window.append(curr_loss)
            if len(avg_loss_window) > 999:
                del avg_loss_window[0]

            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if iteration % 1000 == 999:
                loss_avg = running_loss / iteration
                loss_window = sum(avg_loss_window) / 1000
                acc = 100 * correct / total
                iteration = epoch * len(list(dataset.trainloader)) + i
                writer.add_scalar(
                    "Average Cumulated Loss/Train", loss_avg, iteration,
                )
                writer.add_scalar(
                    "Average Loss Window/Train", loss_window, iteration,
                )
                writer.add_scalar("Accuracy/Train", acc, iteration)
                logging.info("Average Loss:\t%.4f\tAcc:\t%.4f" % (loss_avg, acc))
                iter_list.append(iteration)
                loss_list.append(loss_avg)
                accuracy_list.append(acc)

                running_loss = 0.0
                correct = 0
                total = 0

        logging.info("Finished Epoch %i / %i" % (epoch + 1, config["Setup"]["Epochs"]))

        _epoch_end = time.time()

        _epoch_time = _epoch_end - _epoch_start

        _epoch_process_time += _epoch_time
        logging.info("Time for epoch:\t%8.8f" % (_epoch_time))

    logging.info("Finished Training")
    _global_end = time.time()

    _global_process_time = _global_end - _global_start

    logging.info(
        "Average time for each epoch:\t%8.8f" % (_epoch_process_time / config["Setup"]["Epochs"])
    )
    logging.info("Time needed for training:\t%8.8f" % (_global_process_time))
    if config["Setup"]["Parallel"] == 1:
        PATH = (
            "./data/models/save/"
            + net.module.name
            + "_"
            + dataset.name
            + "_"
            + datetime.now().strftime("%m-%d-%Y-%H:%M:%S")
            + ".pth"
        )
    else:
        PATH = (
            "./data/models/save/"
            + net.name
            + "_"
            + dataset.name
            + "_"
            + datetime.now().strftime("%m-%d-%Y-%H:%M:%S")
            + ".pth"
        )

    torch.save(net.state_dict(), PATH)

    if return_data:
        return net, iter_list, loss_list, accuracy_list
    else:
        return net


def test(net, dataset, return_data=False):

    if config["Setup"]["Parallel"] == 1:
        writer = SummaryWriter("./data/models/log/runs/" + net.module.writer + "/test")
    else:
        writer = SummaryWriter("./data/models/log/runs/" + net.writer + "/test")

    final_correct = 0
    final_total = 0
    final_running_loss = 0.0
    final_cumulated_labels = []
    final_cumulated_predictions = []

    with torch.no_grad():
        for idx, data in enumerate(dataset.testloader):

            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)

            loss = criterion(outputs, labels)
            final_running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)

            final_total += labels.size(0)
            final_correct += (predicted == labels).sum().item()

            final_cumulated_labels.extend(labels.tolist())
            final_cumulated_predictions.extend(predicted.tolist())

    logging.info("Final Metrics")

    logging.info(
        "Loss: %.3f\tAccuracy: %.3f%%\tF_Score: %.3f%%"
        % (
            final_running_loss / (idx + 1),
            100 * final_correct / final_total,
            100
            * f1_score(
                y_true=final_cumulated_labels, y_pred=final_cumulated_predictions, average="macro",
            ),
        )
    )
    return


def production(net, data):
    return net(data)
