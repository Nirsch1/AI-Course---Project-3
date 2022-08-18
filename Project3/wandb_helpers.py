# wandb_helpers
#!pip install wandb
from datetime import datetime
import wandb
from collections import namedtuple
import numpy as np
import os
import tensorflow as tf

Dataset = namedtuple("Dataset", ["images", "labels"])
dataset_names = ["training", "validation", "test"]

def start_wandb_run(model_name, config):
    timestamp = datetime.now().strftime("%H%M%S")
    return wandb.init(project=f"ml-p2", entity="ml-p2", name=f"{model_name}-{timestamp}" , 
        notes = f"Training FCNN model @{timestamp}", config = config)

def read_datasets(wandb_run, dataset_tag = "latest"):
    '''
    Read all datasets from W&B.
    Usage example: train_set, validation_set, test_set = wbh.read_datasets(run)
    '''
    artifact = wandb_run.use_artifact(f'ml-p2/ml-p2/fashion-mnist:{dataset_tag}', type='dataset')
    data_dir = artifact.download()
    return [ read_dataset(data_dir, ds_name) for ds_name in dataset_names ]

def read_dataset(data_dir, ds_name):
    filename = ds_name + ".npz"
    data = np.load(os.path.join(data_dir, filename))
    return Dataset(images = data["x"], labels = data["y"])

def read_model(wandb_run, model_name, model_tag = "latest") -> tf.keras.models.Model:
    artifact = wandb_run.use_artifact(f'ml-p2/ml-p2/{model_name}:{model_tag}', type='model')
    artifact_dir = artifact.download()
    return tf.keras.models.load_model(artifact_dir)

def save_model(wandb_run, model, config, model_name, model_description):
    model_file = f'./saved-models/{model_name}.tf'
    tf.keras.models.save_model(model, model_file)
    model_artifact = wandb.Artifact(model_name, type = "model", description=model_description, metadata= dict(config))
    model_artifact.add_dir(model_file)
    wandb_run.log_artifact(model_artifact)

def load_best_model(sweep_id):
    api = wandb.Api()
    sweep = api.sweep(f"ml-p2/ml-p2/{sweep_id}")
    runs = sorted(sweep.runs,
        key=lambda run: run.summary.get("val_accuracy", 0), reverse=True)
    val_acc = runs[0].summary.get("val_accuracy", 0)
    print(f"Best run {runs[0].name} with {val_acc} validation accuracy")

    model_file = runs[0].file("model-best.h5").download(replace=True)
    model_file.close()

#if (__name__ == "__main__"):
#    load_best_model("6zmewzd0")
