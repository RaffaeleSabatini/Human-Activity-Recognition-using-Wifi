import numpy as np
import numpy.random as rnd

from os import listdir, makedirs, path
from glob import glob
from tqdm import tqdm

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
        print("Datasets are already splitted!")
    
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
                        image_idx = i*4+stream
                        np.save(test_dataset+'/'+trace_name[:-5]+str(image_idx), full_trace[i*doppler_trace_size:(i+1)*doppler_trace_size, :], allow_pickle=True)

        print("Test dataset created successfully!\n")
    else:
        print("Dataset is already splitted!\n")