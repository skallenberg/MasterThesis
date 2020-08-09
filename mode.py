import torch
import torch.nn as nn
import torch.optim as optim
from utils.config import Config
from sklearn.metrics import f1_score
import logging
from utils import visualize, set_config
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

config = Config.get_instance()
criterion = nn.CrossEntropyLoss()

if torch.cuda.is_available():
    device = torch.device("cuda:1")
else:
    device = torch.device("cpu")


def train(net, dataset):

    writer_name = (
        net.name
        + "_"
        + dataset.name
        + "_"
        + datetime.now().strftime("%m-%d-%Y-%H:%M:%S")
    )
    writer = SummaryWriter("./data/models/log/runs/" + writer_name + "/train")

    net.writer = writer_name
    optimizer = set_config.choose_optimizer(net)

    for epoch in range(
        config["Setup"]["Epochs"]
    ):  # loop over the dataset multiple times

        running_loss = 0.0
        correct = 0
        total = 0
        cumulated_labels = []
        cumulated_predictions = []
        for i, data in enumerate(dataset.trainloader, 0):

            if i > 50:
                break

            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = net(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)

            cumulated_labels.extend(labels.tolist())
            cumulated_predictions.extend(predicted.tolist())

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if i % 10 == 9:
                writer.add_scalar(
                    "Loss/Train",
                    running_loss / 10,
                    epoch * len(dataset.trainloader) + i,
                )
                writer.add_scalar(
                    "Accuracy/Train",
                    100 * correct / total,
                    epoch * len(dataset.trainloader) + i,
                )
                writer.add_scalar(
                    "F-Score/Train",
                    100
                    * f1_score(
                        y_true=cumulated_labels,
                        y_pred=cumulated_predictions,
                        average="macro",
                    ),
                    epoch * len(dataset.trainloader) + i,
                )
                # writer.add_figure(
                #    "Predictions vs. Truth",
                #    visualize.plot_classes_preds(net, inputs, labels, dataset.classes),
                #    global_step=epoch * len(dataset.trainloader) + i,
                # )

                running_loss = 0.0
                correct = 0
                total = 0
                cumulated_labels = []
                cumulated_predictions = []
    print("Finished Training")
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
    return net


def test(net, dataset):

    writer = SummaryWriter("./data/models/log/runs/" + net.writer + "/test")

    correct = 0
    total = 0
    num_classes = len(dataset.classes)
    class_correct = list(0.0 for i in range(num_classes))
    class_total = list(0.0 for i in range(num_classes))
    running_loss = 0.0
    cumulated_labels = []
    cumulated_predictions = []

    final_correct = 0
    final_total = 0
    final_running_loss = 0.0
    final_cumulated_labels = []
    final_cumulated_predictions = []

    with torch.no_grad():
        for idx, data in enumerate(dataset.testloader):

            if idx > 10:
                break

            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)

            loss = criterion(outputs, labels)
            running_loss += loss.item()
            final_running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            final_total += labels.size(0)
            final_correct += (predicted == labels).sum().item()

            c = (predicted == labels).squeeze()

            cumulated_labels.extend(labels.tolist())
            cumulated_predictions.extend(predicted.tolist())

            final_cumulated_labels.extend(labels.tolist())
            final_cumulated_predictions.extend(predicted.tolist())

            if idx % 10 == 9:
                writer.add_scalar(
                    "Loss/Test", running_loss / 10, len(dataset.trainloader) + idx
                )
                writer.add_scalar(
                    "Accuracy/Test",
                    100 * correct / total,
                    len(dataset.trainloader) + idx,
                )
                writer.add_scalar(
                    "F-Score/Test",
                    100
                    * f1_score(
                        y_true=cumulated_labels,
                        y_pred=cumulated_predictions,
                        average="macro",
                    ),
                    len(dataset.trainloader) + idx,
                )

                running_loss = 0.0
                correct = 0
                total = 0
                cumulated_labels = []
                cumulated_predictions = []

            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    print("Final Metrics")
    print(
        "Loss: %.3f\tAccuracy: %.3f%%\tF_Score: %.3f%%"
        % (
            final_running_loss / 10,
            100 * final_correct / final_total,
            100
            * f1_score(
                y_true=final_cumulated_labels,
                y_pred=final_cumulated_predictions,
                average="macro",
            ),
        )
    )
    for i in range(num_classes):
        print(
            "Accuracy of %5s : %2d %%"
            % (dataset.classes[i], 100 * class_correct[i] / class_total[i])
        )


def production(net, data):
    return net(data)
