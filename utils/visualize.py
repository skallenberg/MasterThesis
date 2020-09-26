import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.tensorboard import SummaryWriter

from utils import load_data
from utils.config import Config

# matplotlib.use("Qt5Agg")

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")


def rev_normalize(img):
    config = Config().get_instance()

    if config["Setup"]["Data"] == "cifar10":
        std = [0.24703233, 0.24348505, 0.26158768]
        mean = [0.49139968, 0.48215827, 0.44653124]
        img[0, :, :] = img[0, :, :] * std[0] + mean[0]
        img[1, :, :] = img[1, :, :] * std[1] + mean[1]
        img[2, :, :] = img[2, :, :] * std[2] + mean[2]
        return img


def imshow(img, show=True):
    img = rev_normalize(img)
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    if show:
        plt.show()
    return


def show(dataset, nimages=4, net=None, pltshow=True):

    dataiter = iter(dataset.testloader)
    images, labels = dataiter.next()
    images, labels = images.to(device), labels.to(device)

    classes = dataset.classes

    imshow(torchvision.utils.make_grid(images), show=pltshow)
    print("GroundTruth: ", " ".join("%5s" % classes[labels[j]] for j in range(4)))

    if net is not None:
        outputs = net(images)

        _, predicted = torch.max(outputs, 1)

        print("Predicted: ", " ".join("%5s" % classes[predicted[j]] for j in range(4)))
    return


def images_to_probs(net, images):
    """
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    """
    output = net(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]


def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


def plot_classes_preds(net, images, labels, classes):
    """
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    """
    preds, probs = images_to_probs(net, images)
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(12, 48))
    for idx in np.arange(4):
        ax = fig.add_subplot(1, 4, idx + 1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx], one_channel=True)
        ax.set_title(
            "{0}, {1:.1f}%\n(label: {2})".format(
                classes[preds[idx]], probs[idx] * 100.0, classes[labels[idx]]
            ),
            color=("green" if preds[idx] == labels[idx].item() else "red"),
        )
    return fig


def make_graph(net):

    writer = SummaryWriter("./data/models/log/graphs/" + net.name)

    dataset = load_data.get_data()

    pre_iter = iter(dataset.trainloader)
    images, labels = pre_iter.next()
    images, labels = images.to(device), labels.to(device)

    writer.add_graph(net, images)
