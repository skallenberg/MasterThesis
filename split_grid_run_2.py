import pandas as pd
from tqdm import tqdm 

input_list = ["grid_full_2.txt"]

file_list = []
for file in tqdm(input_list):
    print("Reading\t",file)
    with open("grid_results/"+file,"r", encoding="utf-8") as f:
        content = f.read()
    f.close()
    model_list = []
    line_list = []
    for line in tqdm(content.split("\n")):
        line_list.append(line)
        if "Run successfull" in line:
            model_list.append("\n".join(line_list))
            line_list = []
        elif "Run Failed" in line:
            model_list.append("\n".join(line_list))
            line_list = []
    file_list.append(model_list)
    model_list = []
    
result_frame = pd.DataFrame(columns=["run",
                                     "model",
                                     "mode",
                                     "epoch",
                                     "optimizer",
                                     "learnRate",
                                     "mixUp",
                                     "labelSmoothing",
                                     "depthwiseConvolution",
                                     "metric",
                                     "value"])
dtypes = {"run": "int64",
          "model" : "object",
          "mode": "object",
          "epoch": "int64",
          "optimizer" : "object",
          "learnRate" : "float64",
          "mixUp" : "object",
          "labelSmoothing" : "object",
          "depthwiseConvolution" : "object",
          "metric" : "object",
          "value": "float64"
          }
result_frame.astype(dtypes)
metric_list = ["Average Accuracy",
               "Running Average Accuracy",
               "Average Loss",
               "Running Average Loss",
               "Average Precision",
               "Average Recall",
               "Average F1-Score",
               "Average ROC AUC"]

for idx, mlist in tqdm(enumerate(file_list), total=len(file_list)):
    for m in tqdm(mlist):
        model = "NaN"
        epoch = 0
        mode = "NaN"
        mval = 0.0
        metr = "NaN"
        optimizer = "NaN"
        lr = 0.0
        mixup = "NaN"
        labelsmoothing = "NaN"
        dconv = "NaN"
        for line in tqdm(m.split("\n")):
            found_metr = False
            if "Using Model" in line:
                model = line.split("\t")[-1]
            if "Using Optimizer" in line:
                optimizer = line.split("\t")[-1]
            if "Using LearnRate" in line:
                lr = line.split("\t")[-1]
            if "Using MixUp" in line:
                mixup = line.split("\t")[-1]
            if "Using LabelSmoothing" in line:
                labelsmoothing = line.split("\t")[-1]
            if "Using DepthwiseConvolutions" in line:
                dconv = line.split("\t")[-1]
            if "Training Results" in line:
                epoch  = line.split("\t")[-1]
                mode = "train"
            if "Validation Results" in line:
                epoch  = eval(line.split("\t")[-1])
                mode = "val"
            for metric in metric_list:
                if metric in line:
                    mval = eval(line.split("\t")[-1])
                    metr = metric
                    found_metr = True
            if found_metr:
                result_frame = result_frame.append({
                                    "run": idx,
                                    "model" : model,
                                    "mode":mode,
                                    "epoch":epoch,
                                    "optimizer" : optimizer,
                                    "learnRate" : lr,
                                    "mixUp" : mixup,
                                    "labelSmoothing" : labelsmoothing,
                                    "depthwiseConvolution" : dconv,
                                    "metric":metr,
                                    "value": mval}, ignore_index=True)

result_frame.to_csv("grid_results/result_frame_2.csv", index=False)