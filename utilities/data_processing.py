import numpy as np
import numpy.random as rnd
import torch

from os import listdir, makedirs, path
from glob import glob
from sympy import python
from tqdm import tqdm
from torch.utils.data import DataLoader

#----------------------------------------------------------------------------------------------
#------------------------------------  DATASET SPLITTING  -------------------------------------
#----------------------------------------------------------------------------------------------

def create_train_test_split(dataset_path, doppler_trace_size, activity_list,
                             train_ratio=0.7, ds_name="", seed=0):
    """Create train/test datasets with a separate split for every source set.

    ``dataset_path`` may be a glob such as ``doppler_traces/S[1,3]*``.
    Samples are generated from complete doppler windows, shuffled
    deterministically, and split independently for each S-set.
    """
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")

    dataset_root = dataset_path.split('/')[0]
    train_dataset = dataset_root + "_" + ds_name + "train"
    test_dataset = dataset_root + "_" + ds_name + "test"

    if path.exists(train_dataset) or path.exists(test_dataset):
        raise FileExistsError(
            f"Remove or rename existing output directories: {train_dataset}, {test_dataset}"
        )

    makedirs(train_dataset)
    makedirs(test_dataset)
    rng = np.random.default_rng(seed)
    samples_by_set = {}

    subdir_list = sorted([name for name in glob(dataset_path) if path.isdir(name)])
    for subdir in subdir_list:
        source_set = path.basename(subdir)[:2]
        samples_by_set.setdefault(source_set, [])

        for trace_name in tqdm(sorted(listdir(subdir)), desc=f"Reading {subdir}"):
            if not trace_name.endswith(".txt") or trace_name.split('_')[1][0] not in activity_list:
                continue

            stream = int(trace_name[-5])
            full_trace = np.load(path.join(subdir, trace_name), allow_pickle=True)
            n_images = full_trace.shape[0] // doppler_trace_size

            for i in range(n_images):
                image_idx = f"{i * 4 + stream:04d}"
                image_name = trace_name[:-5] + image_idx + ".npy"
                sample = full_trace[
                    i * doppler_trace_size:(i + 1) * doppler_trace_size, :
                ]
                samples_by_set[source_set].append((image_name, sample))

    for source_set, samples in samples_by_set.items():
        rng.shuffle(samples)
        split_idx = int(len(samples) * train_ratio)
        for output_dir, selected_samples in (
            (train_dataset, samples[:split_idx]),
            (test_dataset, samples[split_idx:]),
        ):
            for image_name, sample in selected_samples:
                np.save(path.join(output_dir, image_name[:-4]), sample, allow_pickle=True)

    print(
        f"Created {train_dataset} and {test_dataset} with a "
        f"{train_ratio:.0%}/{1 - train_ratio:.0%} split per source set."
    )

def create_train_dataset(dataset_path, doppler_trace_size, activity_list, ds_name=""):
    train_dataset = dataset_path.split('/')[0] + "_" + ds_name + "train"

    if train_dataset not in listdir():
        print("Creating train dataset...")
        makedirs(train_dataset, exist_ok=True)
        subdir_list = sorted([name for name in glob(dataset_path) if path.isdir(name)])

        # Splitting a trace in many sub-traces and assigning them to train dataset
        for subdir in subdir_list:
            print(f"Splitting {subdir}:")
            for trace_name in tqdm(sorted(listdir(subdir))):
                # Only traces whose label is in ACTIVITIES are used for training/testing
                if trace_name.split('_')[1][0] in activity_list:
                    trace_path = subdir+'/'+trace_name
                    full_trace = np.load(trace_path, allow_pickle=True)
                    N_images = full_trace.shape[0]//doppler_trace_size

                    for i in range(N_images):
                        np.save(train_dataset+'/'+trace_name[:-4]+f"-{i}", full_trace[i*doppler_trace_size:(i+1)*doppler_trace_size, :], allow_pickle=True)

        print("Train dataset created successfully!")
    else:
        print("Dataset is already splitted!\n")
    
#----------------------------------------------------------------------------------------------

def create_test_dataset(dataset_path, doppler_trace_size, activity_list, ds_name=""):
    test_dataset  = dataset_path.split('/')[0] + "_" + ds_name + "test"

    if test_dataset not in listdir():
        print(f"Creating test dataset...")
        makedirs(test_dataset, exist_ok=True)

        subdir_list = sorted([name for name in glob(dataset_path) if path.isdir(name)])
        for subdir in subdir_list:
            print(f"Splitting {subdir}:")

            traces_list = sorted([trace for trace in listdir(subdir) if trace.endswith(".txt")])
            for trace_name in tqdm(traces_list):
                stream     = int(trace_name[-5])
                activity   = trace_name.split('_')[1][0]
                if activity in activity_list: 
                    full_trace = np.load(subdir+'/'+trace_name, allow_pickle=True)
                    N_images   = full_trace.shape[0]//doppler_trace_size

                    for i in range(N_images):
                        # Images relative to same action but measured by different antennas are contiguously labelled in the dataset
                        image_idx = f"{i*4+stream:04d}"
                        np.save(test_dataset+'/'+trace_name[:-5]+image_idx, full_trace[i*doppler_trace_size:(i+1)*doppler_trace_size, :], allow_pickle=True)

        print("Test dataset created successfully!\n")
    else:
        print("Dataset is already splitted!\n")


#---------------------------------------------------------------------------------------------
#------------------------------------  MODEL EVALUATION  -------------------------------------
#---------------------------------------------------------------------------------------------

def compute_metrics(metrics, model, validation_dataset, labels, device, debug=False):
    '''
        Computes different metrics of the model, specified by the parameter 'metrics'.
        Possible choices of the metrics are:
        
        •) confusion matrix (cm) of the model for a given validation dataset. 
        •) precision
        •) recall
        •) f1-score
            (f1-score is the harmonic average of precision and recall)
    '''
    # Block code execution if given metrics cannot be computed/do not exist
    known_metrics = ["cm", "precision", "recall", "f1"]
    if isinstance(metrics, list):
        for m in metrics:
            if m not in known_metrics:
                raise ValueError(f"Error: {m} is not a known metric. Choose one among {known_metrics}.")
    elif isinstance(metrics, str):
        if metrics == "all": 
            metrics = known_metrics
        elif metrics in known_metrics: 
            metrics = [metrics]
        else:
            raise ValueError(f"Error: {metrics} is not a knwon metric!")
    else:
        raise ValueError("metrics must be a string or a list!")

    # Collect counts for each predicted labels in order to handle metrics computations at the end
    model.eval()
    counts_matrix = np.zeros(shape=(len(labels), len(labels)))
    for i in range(len(labels)):
        single_activity_dataset = validation_dataset.retrieve_activity(i)
        single_activity_dataloader = DataLoader(single_activity_dataset, batch_size=128, shuffle=True)

        correct = 0

        for X, Y in tqdm(single_activity_dataloader):
            X = X.to(device)
            Y = Y.to(device)

            with torch.no_grad():
                logits = model(X.unsqueeze(1))
                pred = logits.argmax(dim=1)
            pred_labels, counts = np.unique(pred.cpu().numpy(), return_counts=True)
            counts_matrix[i, pred_labels] += counts

            correct += (pred == Y).float().sum().item()

        if debug:
            print(f"Label: {labels[i]} --- Score: {correct/len(single_activity_dataloader.dataset)}")
            print(f"Counts matrix:\n{counts_matrix}")
    
    # Compute metrics only if present in metrics
    output = {m:None for m in metrics}
    if "cm" in metrics or metrics == "all": 
        confusion_matrix = counts_matrix / (np.sum(counts_matrix, axis=1, keepdims=True) + 1e-9)
        output["cm"]=confusion_matrix

    if "precision" in metrics or metrics == "all":
        precisions = counts_matrix.diagonal() / (np.sum(counts_matrix, axis=0) + 1e-9)
        output["precision"] = precisions

    if "recall" in metrics or metrics == "all":
        recalls = counts_matrix.diagonal() / (np.sum(counts_matrix, axis=1) + 1e-9)
        output["recall"] = recalls

    if "f1" in metrics or metrics == "all":
        if "precision" not in metrics: precisions = counts_matrix.diagonal() / (np.sum(counts_matrix, axis=0) + 1e-9)
        if "recall" not in metrics: recalls = counts_matrix.diagonal() / (np.sum(counts_matrix, axis=1) + 1e-9)
        f1_score = 2*precisions*recalls/(precisions+recalls+1e-9)
        output["f1"] = f1_score

    return output

#---------------------------------------------------------------------------------------------
#--------------------------------------  MODEL SAVING  ---------------------------------------
#---------------------------------------------------------------------------------------------

def save_best_F1_model(model, metrics):
    """
    Save the current model if it has a better mean F1-score than
    the currently stored best model.

    Parameters
    ----------
    model : torch.nn.Module
        Current model.

    metrics : dict
        Dictionary returned by compute_metrics().
        Must contain the "f1" key.
    """

    best_model_dir = "best_model"
    model_path = path.join(best_model_dir, "model.pth")
    metrics_path = path.join(best_model_dir, "metrics.npz")

    # Mean F1-score of the current model
    current_f1_mean = np.mean(metrics["f1"])

    # If no best model exists, save the current one
    if not path.exists(best_model_dir):
        makedirs(best_model_dir)

        torch.save(model, model_path)

        np.savez(
            metrics_path,
            **metrics,
            mean_f1=current_f1_mean
        )

        print(f"Best model saved! Mean F1-score: {current_f1_mean:.4f}")

        return 

    # Load metrics of the previously saved best model
    old_metrics = np.load(metrics_path)
    old_f1_mean = float(old_metrics["mean_f1"])

    # Replace the old model if the current one is better
    if current_f1_mean > old_f1_mean:

        torch.save(model, model_path)

        np.savez(
            metrics_path,
            **metrics,
            mean_f1=current_f1_mean
        )

        print(
            f"New best model saved! "
            f"Mean F1-score: {old_f1_mean:.4f} -> {current_f1_mean:.4f}"
        )

        return 

    else:
        print(
            f"Model not saved. "
            f"Mean F1-score: {current_f1_mean:.4f} "
            f"(best: {old_f1_mean:.4f})"
        )

        return 

