import os
import sys
# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import gc
import lightning.pytorch as pl
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from datasets import Dataset,load_dataset,load_from_disk
from concurrent.futures import ThreadPoolExecutor


from caduceus.tokenization_caduceus import CaduceusTokenizer

from src.dataloaders.datasets.hg38_dataset import HG38Dataset
from src.dataloaders.utils.utils import download_sequences_chunks,split_sequences,unzip_s3_zip,load_fasta
from src.dataloaders.utils.mlm import mlm_getitem
from src.dataloaders.utils.rc import coin_flip, string_reverse_complement


class CustomMultiSpeciesDataset(torch.utils.data.Dataset):
    """Dataset to loop through sequences, tokenize them, and provide fixed-size chunks."""

    def __init__(
            self,
            data_source,
            max_length,
            mlm=False,
            mlm_probability=0.15,
            pad_max_length=None,
            tokenizer=None,
            add_eos=False,
            rc_aug=False,
            num_workers=1,
    ):
        self.data_source = data_source
        self.mlm = mlm
        self.mlm_probability = mlm_probability
        self.max_length = max_length
        self.pad_max_length = pad_max_length if pad_max_length is not None else max_length
        self.tokenizer = tokenizer
        self.add_eos = add_eos
        self.rc_aug = rc_aug
        self.num_workers = num_workers

        # Parallel processing of sequences
        with ThreadPoolExecutor(max_workers=1) as executor:
            self.tokenized_chunks = []
            counter = 0
            total = len(data_source)
            for result in executor.map(self.process_sequence, data_source):
                self.tokenized_chunks.extend(result)
                print(f"Processed {counter} sequences out of {total}.", end="\r")

        assert len(self.tokenized_chunks) > 0, "No sequences were processed."
        del self.data_source
    

    

    def process_sequence(self, sequence):
        """Processes a single sequence by augmenting, tokenizing, and chunking it."""
        if self.rc_aug and coin_flip():
            sequence = string_reverse_complement(sequence)
        tokens = self.tokenizer(sequence, add_special_tokens=False)['input_ids']
        chunk_size = self.max_length - (1 if self.add_eos else 0)  # Adjust for EOS token if needed
        return [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]
    
    def __len__(self):
        return len(self.tokenized_chunks)

    def __getitem__(self, idx):
        """Return a chunk with optional end-of-sequence token and masked language modeling."""
        tokenized_sequence = self.tokenized_chunks[idx]

        if self.add_eos:
            tokenized_sequence.append(self.tokenizer.sep_token_id)

        if len(tokenized_sequence) < self.pad_max_length:
            tokenized_sequence += [self.tokenizer.pad_token_id] * (self.pad_max_length - len(tokenized_sequence))

        tokenized_sequence = torch.tensor(tokenized_sequence, dtype=torch.long)

        if self.mlm:
            data, target = mlm_getitem(
                tokenized_sequence,
                mlm_probability=self.mlm_probability,
                contains_eos=self.add_eos,
                tokenizer=self.tokenizer,
                eligible_replacements=None,
            )
        else:
            data = tokenized_sequence[:-1].clone()
            target = tokenized_sequence[1:].clone()

        return data, target
class DynamicMultiSpeciesDataset(IterableDataset):
    """IterableDataset to load sequences dynamically with reduced memory usage."""

    def __init__(
        self,
        data_source,
        max_length,
        first_chunk,
        last_chunk,
        mlm=False,
        mlm_probability=0.15,
        pad_max_length=None,
        tokenizer=None,
        add_eos=False,
        rc_aug=False,
    ):
        self.data_source = data_source
        self.mlm = mlm
        self.mlm_probability = mlm_probability
        self.max_length = max_length
        self.pad_max_length = pad_max_length if pad_max_length is not None else max_length
        self.tokenizer = tokenizer
        self.add_eos = add_eos
        self.rc_aug = rc_aug
        self.first_chunk = first_chunk
        self.last_chunk = last_chunk

    def _process_sequence(self, sequence):
        """Processes a single sequence by augmenting, tokenizing, and chunking it."""
        if self.rc_aug and coin_flip():
            sequence = string_reverse_complement(sequence)

        tokens = self.tokenizer(sequence, add_special_tokens=False)['input_ids']
        chunk_size = self.max_length - (1 if self.add_eos else 0)
        chunks = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]
        return chunks

    def __iter__(self):

        worker_info = get_worker_info()
        if worker_info is None:  # Single-process data loading
            first_chunk = self.first_chunk
            last_chunk = self.last_chunk
        else:  # In a worker process
            # Split the chunks evenly among the workers
            per_worker = (self.last_chunk - self.first_chunk) // worker_info.num_workers
            worker_id = worker_info.id
            first_chunk = self.first_chunk + worker_id * per_worker
            last_chunk = first_chunk + per_worker
            # Ensure the last worker gets any remaining chunks
            if worker_id == worker_info.num_workers - 1:
                last_chunk = self.last_chunk

        for num_chunk in range(first_chunk, last_chunk):
            chunk_dataset_path = self.data_source.replace("*", str(num_chunk))
            chunk_sequences = load_fasta(chunk_dataset_path)

            for sequence in chunk_sequences:
                tokenized_chunks = self._process_sequence(sequence)
                for chunk in tokenized_chunks:
                    tokenized_sequence = chunk

                    if self.add_eos:
                        tokenized_sequence.append(self.tokenizer.sep_token_id)

                    if len(tokenized_sequence) < self.pad_max_length:
                        tokenized_sequence += [self.tokenizer.pad_token_id] * (self.pad_max_length - len(tokenized_sequence))

                    tokenized_sequence = torch.tensor(tokenized_sequence, dtype=torch.long)
                    
                    if self.mlm:
                        data, target = mlm_getitem(
                            tokenized_sequence,
                            mlm_probability=self.mlm_probability,
                            contains_eos=self.add_eos,
                            tokenizer=self.tokenizer,
                            eligible_replacements=None,
                        )
                    else:
                        data = tokenized_sequence[:-1].clone()
                        target = tokenized_sequence[1:].clone()

                    yield data, target
                del tokenized_chunks
                del sequence
            del chunk_sequences
            gc.collect()


class LitDynamicMultiSpecies(pl.LightningDataModule):
    def __init__(self, dataset_cfg):
        super().__init__()

        self.__name__ = dataset_cfg["_name_"]
        self.dataset_path = dataset_cfg["dataset_path"]
        self.tokenizer_name = dataset_cfg["tokenizer_name"]
        self.rc_aug = dataset_cfg["rc_aug"]
        self.max_length = dataset_cfg["max_length"]
        self.max_length_val = dataset_cfg["max_length_val"] if dataset_cfg["max_length_val"] is not None else dataset_cfg["max_length"]
        self.max_length_test = dataset_cfg["max_length_test"] if dataset_cfg["max_length_test"] is not None else dataset_cfg["max_length"]
        self.batch_size = dataset_cfg["batch_size"]
        self.add_eos = dataset_cfg["add_eos"]
        self.batch_size_eval = dataset_cfg["batch_size_eval"] if dataset_cfg["batch_size_eval"] is not None else self.batch_size
        self.num_workers = dataset_cfg["num_workers"]
        self.num_chunks= dataset_cfg["num_chunks"] 

        self.mlm = dataset_cfg["mlm"]
        self.mlm_probability = dataset_cfg["mlm_probability"]

        self.tokenizer = CaduceusTokenizer(
            model_max_length=1e7,
            add_special_tokens=False
        )
        self.vocab_size = 0

    def prepare_data(self):

        splitting_chunk = int(self.num_chunks * 0.99)

        self.dataset_train, self.dataset_val = [
            DynamicMultiSpeciesDataset(data_source=self.dataset_path,
                                       max_length=max_len,
                                        first_chunk=first,
                                        last_chunk=last,
                                       tokenizer=self.tokenizer,
                                       add_eos=self.add_eos,
                                       rc_aug=self.rc_aug,
                                       mlm=self.mlm,
                                       mlm_probability=self.mlm_probability, )
            for (first,last), max_len in zip([(0,splitting_chunk), (splitting_chunk,self.num_chunks)], [self.max_length, self.max_length_val])
        ]
        gc.collect()
        print(f"Finished preparing datasets . Train chunks: from {self.dataset_train.first_chunk} to {self.dataset_train.last_chunk -1}, Val chunks: from chunk {self.dataset_val.first_chunk} to {self.dataset_val.last_chunk -1}")


    def setup(self, stage=None):
        assert self.dataset_train is not None and self.dataset_val is not None
        self.vocab_size = len(self.tokenizer)
        print("Set up of the Data Module is successful")

    def train_dataloader(self, **kwargs) -> DataLoader:
        loader = DataLoader(self.dataset_train, batch_size=self.batch_size, pin_memory=False, num_workers=self.num_workers, **kwargs)
        return loader

    def val_dataloader(self, **kwargs) -> DataLoader:
        kwargs["drop_last"] = False
        return DataLoader(self.dataset_val, batch_size=self.batch_size_eval, pin_memory=False, num_workers=self.num_workers, **kwargs)


class LitHG38(pl.LightningDataModule):
    def __init__(self,dataset_cfg):
        super().__init__()

        self.__name__=dataset_cfg["_name_"]
        self.tokenizer_name = dataset_cfg["tokenizer_name"]
        self.rc_aug = dataset_cfg["rc_aug"]  # reverse compliment augmentation
        self.max_length = dataset_cfg["max_length"]
        self.max_length_val = dataset_cfg["max_length_val"] if dataset_cfg["max_length_val"] is not None else dataset_cfg["max_length"]
        self.max_length_test = dataset_cfg["max_length_test"] if dataset_cfg["max_length_test"] is not None else dataset_cfg["max_length"]
        self.batch_size = dataset_cfg["batch_size"]
        self.add_eos = dataset_cfg["add_eos"]
        self.batch_size_eval = dataset_cfg["batch_size_eval"] if dataset_cfg["batch_size_eval"] is not None else self.batch_size
        self.shuffle = dataset_cfg["shuffle"]
        self.num_workers = dataset_cfg["num_workers"]
        self.bed_file = dataset_cfg["bed_file"]
        self.fasta_file = dataset_cfg["fasta_file"]

        self.mlm = dataset_cfg["mlm"]
        self.mlm_probability = dataset_cfg["mlm_probability"]
        self.local = dataset_cfg["local"]

        # To be instantiated in `setup`
        self.tokenizer =CaduceusTokenizer(
                model_max_length=self.max_length,
                add_special_tokens=False
            )
        self.vocab_size = 0
    
    def prepare_data(self):
        self.dataset_train, self.dataset_val, self.dataset_test = [
            HG38Dataset(split=split,
                        bed_file=self.bed_file,
                        fasta_file=self.fasta_file,
                        max_length=max_len,
                        tokenizer=self.tokenizer,  # pass the tokenize wrapper
                        tokenizer_name=self.tokenizer_name,
                        add_eos=self.add_eos,
                        return_seq_indices=False,
                        rc_aug=self.rc_aug,
                        return_augs=False,
                        mlm=self.mlm,
                        mlm_probability=self.mlm_probability,
                        local=self.local,)
            for split, max_len in
            zip(["train", "valid", "test"], [self.max_length, self.max_length_val, self.max_length_test])
        ]

    
    def setup(self, stage=None):
       
       self.vocab_size = len(self.tokenizer)
    
    def train_dataloader(self,**kwargs) -> DataLoader:
        """ The train dataloader """
    
        shuffle = self.shuffle
        sampler = None
        loader = self._data_loader(self.dataset_train, batch_size=self.batch_size,
                                   shuffle=shuffle, sampler=sampler,pin_memory=True, num_workers=self.num_workers,**kwargs)
        return loader
    def val_dataloader(self, **kwargs):
        """ The val dataloader """
        kwargs["drop_last"] = False
        return self._data_loader(self.dataset_val, batch_size=self.batch_size_eval,pin_memory=True, num_workers=self.num_workers,**kwargs)
    
    def test_dataloader(self, **kwargs):
        """ The test dataloader """
        kwargs["drop_last"] = False
        # TODO: Should have separate train and eval loaders
        return self._data_loader(self.dataset_test, batch_size=self.batch_size_eval,pin_memory=True,num_workers=self.num_workers, **kwargs)
    
    @staticmethod
    def _data_loader(dataset: Dataset, batch_size: int, shuffle: bool = False, sampler=None, num_workers = 1, **kwargs) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            **kwargs,
        )


class LitMultiSpecies(pl.LightningDataModule):
    def __init__(self,dataset_cfg):
        super().__init__()

        self.__name__=dataset_cfg["_name_"]
        self.dataset_path = dataset_cfg["dataset_path"]
        self.tokenizer_name = dataset_cfg["tokenizer_name"]
        self.rc_aug = dataset_cfg["rc_aug"]  # reverse compliment augmentation
        self.max_length = dataset_cfg["max_length"]
        self.max_length_val = dataset_cfg["max_length_val"] if dataset_cfg["max_length_val"] is not None else dataset_cfg["max_length"]
        self.max_length_test = dataset_cfg["max_length_test"] if dataset_cfg["max_length_test"] is not None else dataset_cfg["max_length"]
        self.batch_size = dataset_cfg["batch_size"]
        self.add_eos = dataset_cfg["add_eos"]
        self.batch_size_eval = dataset_cfg["batch_size_eval"] if dataset_cfg["batch_size_eval"] is not None else self.batch_size
        self.shuffle = dataset_cfg["shuffle"]
        self.num_workers = dataset_cfg["num_workers"]     

        self.mlm = dataset_cfg["mlm"]
        self.mlm_probability = dataset_cfg["mlm_probability"]

        self.tokenizer =CaduceusTokenizer(
                model_max_length=1e7,
                add_special_tokens=False
            )
        self.vocab_size = 0
    
    def prepare_data(self):

        sequences = download_sequences_chunks(
        dataset_path=self.dataset_path,
        worker_id=0,
        num_chunks_per_worker=1024,
        save_to_disk=False,
    )
        print(
            f"Downloaded {len(sequences)} sequences and a total of {sum(len(sublist) for sublist in sequences)} characters."
        )


        train_sequences, validation_sequences = split_sequences(
        sequences, train_proportion=0.99, seed=0
        )

        print("Finished splitting sequences into train and validation sets.")
    
        self.dataset_train, self.dataset_val = [
            CustomMultiSpeciesDataset(data_source=source,
                        max_length=max_len,
                        tokenizer=self.tokenizer,  # pass the tokenize wrapper
                        add_eos=self.add_eos,
                        rc_aug=self.rc_aug,
                        mlm=self.mlm,
                        mlm_probability=self.mlm_probability,
                        num_workers=self.num_workers,)
                for source, max_len in
            zip([train_sequences, validation_sequences], [self.max_length, self.max_length_val])
        ]
        del sequences
        print("Finished preparing datasets.")
    def setup(self, stage=None):
       assert self.dataset_train is not None and self.dataset_val is not None
       self.vocab_size = len(self.tokenizer)
       print("Set up of the Data Module is successful")
    
    def train_dataloader(self,**kwargs) -> DataLoader:
        """ The train dataloader """
    
        shuffle = self.shuffle
        sampler = None
        loader = self._data_loader(self.dataset_train, batch_size=self.batch_size,
                                   shuffle=shuffle, sampler=sampler,pin_memory=True, num_workers=self.num_workers,**kwargs)
        return loader
    def val_dataloader(self, **kwargs)-> DataLoader:
        """ The val dataloader """
        kwargs["drop_last"] = False
        return self._data_loader(self.dataset_val, batch_size=self.batch_size_eval,pin_memory=True, num_workers=self.num_workers,**kwargs)
    
    def test_dataloader(self, **kwargs):
        """ The test dataloader """
        kwargs["drop_last"] = False
        # TODO: Should have separate train and eval loaders
        return self._data_loader(self.dataset_test, batch_size=self.batch_size_eval,pin_memory=True,num_workers=self.num_workers, **kwargs)
    
    
    @staticmethod
    def _data_loader(dataset: Dataset, batch_size: int, shuffle: bool = False, sampler=None, num_workers = 1, **kwargs) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            **kwargs,
        )