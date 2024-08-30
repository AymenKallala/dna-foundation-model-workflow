import hashlib
from typing import List,Tuple
import os
import shutil
from Bio import SeqIO
import fsspec
import numpy as np
import joblib
import s3fs
import zipfile


DEFAULT_CACHE_DIR = "~/.cache"


def read_fasta(filename: str) -> List[str]:
    """
    Reads one fasta file.

    Args:
        filename: Name of the fasta file to be read

    Returns:
        List: List with all sequences.
    """
    return [str(record.seq).upper() for record in SeqIO.parse(filename, "fasta")]

def load_fasta(filepath: str) -> List[str]:
        """
        Downloads and reads one fasta file.

        Args:
            filepath: Path to the fasta file to be read.

        Returns:
            List: List with all sequences.
        """
        print(f"Downloading file {filepath}")

        with fsspec.open(filepath,"r") as file:
            return read_fasta(file)

def split_sequences(
    sequences: List[str],
    train_proportion: float = 0.8,
    seed: int = 0,
) -> Tuple[List[str], List[str]]:
    """
    Splits sequences into train and test sequences.


    Args:
        sequences: List of sequences.
        train_proportion: Training set proportion. Defaults to 0.8.
        seed: Seed. Defaults to 0.

    Returns:
        Training sequences.
        Validation sequences.
    """
    num_train_sequences = int(np.round(train_proportion * len(sequences)))

    random_key = np.random.default_rng(seed)
    indices = np.arange(len(sequences))
    random_key.shuffle(indices)

    train_indices = indices[:num_train_sequences]
    val_indices = indices[num_train_sequences:]

    train_sequences = [sequences[i] for i in train_indices]
    val_sequences = [sequences[i] for i in val_indices]

    return train_sequences, val_sequences



def _get_dir() -> str:
    """
    Get directory to save files on user machine.
    """
    return os.path.expanduser(DEFAULT_CACHE_DIR
    )


def get_hash(arguments: List[str]) -> int:
    """
    Compute a hash based on a list of str arguments.
    """
    m = hashlib.blake2s(digest_size=16)
    for s in arguments:
        if s is not None:
            m.update(s.encode())
    return int(m.hexdigest(), 16)


def download_sequences_chunks(
    dataset_path: str,
    worker_id: int,
    num_chunks_per_worker: int,
    save_to_disk: bool = False,
) -> List[str]:
    """
    Downloads the sequences corresponding to the worker. It will be called separately
    by each worker with their own worker id.

    Args:
        dataset_path: Path to the dataset (on the bucket).
        worker_id: Worker id.
        num_chunks_per_worker: Number of chunks to download for one worker.
        bucket_fasta_handler: Bucket fasta handler to connect to the bucket.
        save_to_disk: Whether to save the resulting files to disk.

    Returns:
        Sequences for this worker.
    """
    # compute save
    hash_dir = hex(
        get_hash([str(worker_id), str(num_chunks_per_worker), str(dataset_path)])
    )
    save_dir = os.path.join(_get_dir(), hash_dir)
    samples_dir = os.path.join(save_dir, "sequences.joblib")

    # check if samples have been previously stored there, if yes load them
    if os.path.exists(samples_dir):
        with open(samples_dir, "rb") as f:
            sequences: List[str] = joblib.load(f)

    else:
        sequences = []

        # get first and last chunk for this worker
        first_chunk = worker_id * num_chunks_per_worker
        last_chunk = int((worker_id + 1) * num_chunks_per_worker)

        for num_chunk in range(first_chunk, last_chunk):
            chunk_dataset_path = dataset_path.replace("*", str(num_chunk))
            chunk_sequences = load_fasta(chunk_dataset_path)
            sequences.extend(chunk_sequences)

        if save_to_disk:
            os.makedirs(save_dir, exist_ok=False)
            print(f"Saving sequences to {samples_dir}")
            with open(samples_dir, "wb") as f:
                joblib.dump(sequences, f)

    return sequences

def save_to_disk_zip(zip_path, extract_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref2:
        zip_ref2.extractall(extract_path)

def gather_folders_content(folder_list, destination_folder):
    """
    Gathers the contents of the given list of folders and moves them into the destination folder.
    
    :param folder_list: List of folders whose contents you want to gather.
    :param destination_folder: The folder where you want to move the contents.
    """
    # Ensure the destination folder exists
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    
    # Iterate over each folder in the list
    for folder in folder_list:
        if os.path.exists(folder) and os.path.isdir(folder):
            # Iterate over each file and directory in the current folder
            for item in os.listdir(folder):
                item_path = os.path.join(folder, item)
                dest_path = os.path.join(destination_folder, item)
                
                try:
                    # Move the item to the destination folder
                    if os.path.isfile(item_path):
                        shutil.move(item_path, dest_path)
                    elif os.path.isdir(item_path):
                        # If it's a directory, move it as well
                        if not os.path.exists(dest_path):
                            shutil.move(item_path, dest_path)
                        else:
                            # Handle conflicts if the directory already exists
                            dest_path = os.path.join(destination_folder, f"{item}_copy")
                            shutil.move(item_path, dest_path)
                    
                except Exception as e:
                    print(f"Error moving {item_path} to {dest_path}: {e}")
        else:
            print(f"Folder {folder} does not exist or is not a directory.")

def unzip_s3_zip(s3_zip_path, extract_path):
    subfolders_paths = []
    s3 = s3fs.S3FileSystem(client_kwargs={"endpoint_url": os.environ.get("S3_ENDPOINT")})
    with s3.open(s3_zip_path, 'rb') as f:
        with zipfile.ZipFile(f, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
            files = zip_ref.namelist()
            for file in files:
                print("Extracting: ", file)
                if file.endswith('.zip'):
                    nested_zip_path = os.path.join(extract_path, file)
                    epcp = extract_path
                    extract_path = os.path.join(extract_path, file.split(".")[0])
                    save_to_disk_zip(nested_zip_path, extract_path)
                    unzipped_folder_path = os.path.join(extract_path,"Full multispecies dataset")

                    if os.path.exists(unzipped_folder_path) and os.path.isdir(unzipped_folder_path):
                        print(f"Unzipped folder : {unzipped_folder_path}")
                        subfolders_paths.append(unzipped_folder_path)

                    extract_path = epcp

    gather_folders_content(subfolders_paths, extract_path)

def compute_total_tokens(dataloader):
    total_tokens = 0
    for data, _ in dataloader:
        total_tokens += data.numel()  # assuming data is your tensor of token IDs
    return total_tokens
