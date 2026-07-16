import numpy as np
import numpy.random as rnd

from os import listdir, makedirs, path
from glob import glob
from tqdm import tqdm
from torch.utils.data import DataLoader

#----------------------------------------------------------------------------------------------
#------------------------------------  DATASET SPLITTING  -------------------------------------
#----------------------------------------------------------------------------------------------

def create_train_dataset(dataset_path, doppler_trace_size, activity_list):
    train_dataset = dataset_path.split('/')[0] + "_train"

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

def create_test_dataset(dataset_path, doppler_trace_size, activity_list):
    test_dataset  = dataset_path.split('/')[0] + "_test"

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

def compute_confusion_matrix(model, validation_dataset, labels, device, debug=False):
    

    model.eval()
    confusion_matrix = np.zeros(shape=(len(labels), len(labels)))
    for i in range(len(labels)):
        single_activity_dataset = validation_dataset.retrieve_activity(i)
        single_activity_dataloader = DataLoader(single_activity_dataset, batch_size=128, shuffle=True)

        correct = 0

        for X, Y in tqdm(single_activity_dataloader):
            X = X.to(device)
            Y = Y.to(device)
            
            pred, logits = model(X.unsqueeze(1), return_pred=True)
            pred_labels, counts = np.unique(pred.cpu().numpy(), return_counts=True)
            confusion_matrix[i, pred_labels] += counts

            correct += (pred == Y).float().sum().item()

        confusion_matrix[i, :] /= np.sum(confusion_matrix[i, :])
        if debug:
            print(f"Label: {labels[i]} --- Score: {correct/len(single_activity_dataloader.dataset)}")
            print(f"Confusion matrix:\n{confusion_matrix}")
        
    return confusion_matrix