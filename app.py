import os
import random
from collections import OrderedDict

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image

import mode
from utils import load_data
from utils import set_config
from utils import visualize
from utils.config import Config

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

st.set_option("deprecation.showfileUploaderEncoding", False)

config = Config().get_instance()

st.title("Master Thesis WebApp")


data_option = st.sidebar.selectbox("Choose Dataset", ["cifar10", "cifar100"],)

config["Setup"]["Data"] = data_option

train_or_test = st.sidebar.selectbox(
    "Choose what you want to do", ["Train new model", "Apply existing model"]
)

model_option = st.sidebar.selectbox(
    "Choose Model to apply",
    ["BaseNet34", "ResNet34", "DenseNet34", "FractalNet4", "NiN_Net34", "RoR_NeXt50", "MG16",],
)

config["Setup"]["Architecture"] = model_option


if train_or_test == "Train new model":
    running_mode = 1

    if torch.cuda.is_available():
        device = torch.device(config["Setup"]["Device"])
        st.sidebar.markdown("Running on the GPU")
    else:
        device = torch.device("cpu")
        st.sidebar.markdown("Running on the CPU")

    epoch_option = st.sidebar.slider("Choose number of epochs", 1, 5, 1)

    optimizer_option = st.sidebar.selectbox("Choose Optimizer", ["SDG", "Adam"],)

    learnrate_option = st.sidebar.slider("Choose LearnRate", 0.001, 0.5, 0.005)

    config["Setup"]["Epochs"] = epoch_option
    config["Optimizer"]["Type"] = optimizer_option
    config["Optimizer"]["LearnRate"] = learnrate_option
else:
    running_mode = 0
    device = torch.device("cpu")
    save_option = st.sidebar.selectbox(
        "Choose saved Model to load",
        [
            i
            for i in list(os.listdir("./data/models/save/"))
            if model_option in i and data_option in i
        ],
    )


config["Setup"]["Architecture"] = model_option


@st.cache
def _load_dataset():
    return load_data.get_data()


def _set_net():
    model = set_config.choose_architecture()
    if running_mode == 1 and torch.cuda.is_available():
        model = nn.DistributedDataParallel(model)
    return model.to(device)


def _load_net_from_file():
    model = _set_net()
    state_dict = torch.load("./data/models/save/" + save_option)
    try:
        model.load_state_dict(state_dict)
    except:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        # load params
        model.load_state_dict(new_state_dict)
    model.eval()
    return model


def _next_input_image(dataiter, size):
    r = random.randint(0, size)
    for i in range(r):
        item = dataiter.next()
    return item


def _train(net, data):
    return mode.train(net, data, return_data=True)


def _test(net, data):
    return mode.test(net, data, return_data=True)


def _visualize(net, data):
    return visualize.show(data, net=net)


if running_mode == 1:
    if st.button("Run"):

        data = _load_dataset()

        st.write("Loaded Data")

        net = _set_net()

        st.write("Set Up Network Architecture")

        net, iterations, losslist, acc, f1 = _train(net, data)

        st.write("Finished Training")

        df = pd.DataFrame(columns=["Loss", "Accuracy", "F1_Score"])
        df["Loss"] = losslist
        df["Accuracy"] = acc
        df["F1_Score"] = f1

        st.write("Training Results")
        st.line_chart(df)

        iterations, losslist, acc, f1 = _test(net, data)

        st.write("Finished Testing")

        df = pd.DataFrame(index=iterations, columns=["Loss", "Accuracy", "F1_Score"])
        df["Loss"] = losslist
        df["Accuracy"] = acc
        df["F1_Score"] = f1

        st.write("Test Results")
        st.line_chart(df)

else:
    if st.button("Classify Image From Dataset"):
        config["DataLoader"]["TestSize"] = 4
        data = _load_dataset()

        images, labels = _next_input_image(iter(data.testloader), len(data.testloader))
        visualize.imshow(torchvision.utils.make_grid(images), show=False)
        st.pyplot()

        classes = data.classes

        st.write("GroundTruth: \t", "\t".join("%5s" % classes[labels[j]] for j in range(4)))

        net = _load_net_from_file()

        outputs = net(images)

        _, predicted = torch.max(outputs, 1)

        st.write("Predicted: \t", "\t".join("%5s" % classes[predicted[j]] for j in range(4)))

    if st.button("Upload Image to classify"):
        data = _load_dataset()
        uploaded_file = st.file_uploader("Choose an image...", type="jpeg")

        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("")
        st.write("Classifying...")
        loader = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
        image = loader(image).float()
        image = image.unsqueeze(0)
        net = _load_net_from_file()
        outputs = net(image)

        _, predicted = torch.max(outputs, 1)
        classes = data.classes
        st.write(classes[predicted])
