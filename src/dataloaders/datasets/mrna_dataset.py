
from pathlib import Path
import torch
from random import randrange, random
import pandas as pd
from Bio import SeqIO


class TranscriptDataset(torch.utils.data.Dataset):
    """
    A Dataset to read mRNA sequences stored in a FASTA file.
    """
    def __init__(self, fasta_file, max_length, pad_max_length=None, tokenizer=None, tokenizer_name=None, add_eos=False, use_padding=True):
        self.max_length = max_length
        self.pad_max_length = pad_max_length if pad_max_length is not None else max_length
        self.tokenizer_name = tokenizer_name
        self.tokenizer = tokenizer
        self.add_eos = add_eos
        self.use_padding = use_padding
        
        fasta_path = Path(fasta_file)
        assert fasta_path.exists(), 'Path to FASTA file must exist'
        
        # Read the FASTA file
        self.sequences = list(SeqIO.parse(fasta_file, "fasta"))

        self.chunk_indices = self.precompute_indices()


    def __len__(self):
        return len(self.chunk_indices)

    def replace_value(self, x, old_value, new_value):
        return torch.where(x == old_value, new_value, x)
    
    def find_end_index(self, start_idx):
        total_length = 0
        curr_idx = start_idx
        

        while curr_idx < len(self.sequences) and total_length + len(self.sequences[curr_idx].seq) + 2 <= self.max_length:
            total_length += len(self.sequences[curr_idx].seq) + 2  # +2 for the BOS/EOS tokens
            curr_idx += 1

        
        if total_length == 0: # the first sequence is longer than max_length
            end_idx = curr_idx
        else:
            end_idx = curr_idx - 1
            
        return end_idx  # Return the last index included in the chunk and the total length

    def precompute_indices(self):
        chunk_indices = []
        start_idx = 0
        
        while start_idx < len(self.sequences):
            end_idx = self.find_end_index(start_idx)
            chunk_indices.append((start_idx, end_idx))
            start_idx = end_idx + 1
            
        return chunk_indices
    
    def get_concatenated_sequences(self, start_idx, end_idx):
        """
        Retrieve sequences from the list between start_idx and end_idx, concatenate them using the EOS token.
        """
        concatenated_seq = []

        for idx in range(start_idx, end_idx + 1):
            current_sequence = str(self.sequences[idx].seq)
            current_sequence = self.tokenizer.bos_token + current_sequence + self.tokenizer.eos_token
            
            concatenated_seq.append(current_sequence)

        return ''.join(concatenated_seq)


    def __getitem__(self, idx):
        """Returns a sequence of specified len"""
        # Get the sequence from 'cdna sequences' column
        start_idx, end_idx = self.chunk_indices[idx]
        seq = self.get_concatenated_sequences(start_idx, end_idx)


        if self.tokenizer_name == 'char':
            seq = self.tokenizer(seq,
                padding="max_length" if self.use_padding else 'do_not_pad',
                max_length=self.pad_max_length,
                truncation=True,
                add_special_tokens=False)  # add cls and eos token (+2)
            seq = seq["input_ids"]  # get input_ids
            #if self.add_eos:
            #    seq.append(self.tokenizer.sep_token_id)  # append list seems to be faster than append tensor

        seq = torch.LongTensor(seq)
        seq = self.replace_value(seq, self.tokenizer._vocab_str_to_int['N'], self.tokenizer.pad_token_id)

        data = seq[:-1].clone()  # remove eos
        target = seq[1:].clone()  # offset by 1, includes eos

        return data, target