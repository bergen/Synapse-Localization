import torch
import pandas as pd
from Bio import SeqIO
from pathlib import Path
import math
import re

class MutationScoring(torch.utils.data.Dataset):

    def __init__(
        self,
        csv_path,
        fasta_path,
        max_length,
        d_output=2, # default binary classification
        tokenizer=None,
        tokenizer_name=None,
        use_padding=True,
        add_eos=False,
    ):
        self.max_length = max_length
        self.tokenizer_name = tokenizer_name
        self.tokenizer = tokenizer
        self.add_eos = add_eos
        self.d_output = d_output  # needed for decoder to grab
        self.use_padding = use_padding

        # Load sequences
        # Find the transcript name in each header
        pattern = re.compile(r'^EN[A-Z]*ST\d+$')
        self.headers_seqs = [(record.description.split("|"), str(record.seq)) for record in SeqIO.parse(fasta_path, "fasta")]
        self.sequence_ids = [self.get_transcript_id(s[0], pattern) for s in self.headers_seqs]
        self.sequences = [x[1] for x in self.headers_seqs]
        self.sequences = dict(zip(self.sequence_ids, self.sequences))



        self.data = pd.read_csv(csv_path)

        # Ensure gene names match between the label file and the fasta file
        self.gene_names = list(set(self.data['transcript_id']) & set(self.sequences.keys()))
        
        # Filter the label DataFrame to keep only the relevant gene names
        self.data = self.data[self.data['transcript_id'].isin(self.gene_names)]


    def __len__(self):
        return len(self.data)
    
    def get_transcript_id(self, header, pattern):
        matches = filter(pattern.match, header)
        try:
            return next(matches)
        except Exception as e:
            return "None"

    def __getitem__(self, idx):
        example = self.data.iloc[idx]
        transcript_id = example['transcript_id']
        deletion_index_start = example['deletion_index_start']
        deletion_index_end = example['deletion_index_end']
        deletion_type = example['deletion_type']

        transcript = self.sequences[transcript_id]

        if deletion_type=='deletion':
            transcript = transcript[:deletion_index_start] + transcript[deletion_index_end:]
        else:
            pass

        seq = self.tokenizer.bos_token + transcript + self.tokenizer.eos_token
        seq = self.tokenizer(
            seq,
            add_special_tokens=False,
            padding="max_length" if self.use_padding else 'do_not_pad',
            max_length=self.max_length+2, # allow for bos and eos tokens
            truncation=True,
        )  
        seq = seq["input_ids"]  # get input_ids

        # convert to tensor
        seq = torch.LongTensor(seq)  

        #return a constant label
        return seq, torch.LongTensor([1]), {'transcript_id':transcript_id, 'deletion_index_start': str(deletion_index_start), 
                                            'deletion_index_end': str(deletion_index_end), 'deletion_type': deletion_type}
