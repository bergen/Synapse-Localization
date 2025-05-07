import torch
import pandas as pd
from Bio import SeqIO
from pathlib import Path
import math
import re

class UTRClassificationDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        csv_path,
        fasta_path,
        synapse_label_name,
        max_length,
        d_output=2, # default binary classification
        dest_path=None,
        tokenizer=None,
        tokenizer_name=None,
        use_padding=True,
        add_eos=False,
        upsample=False
    ):
        self.synapse_label_name = synapse_label_name
        self.max_length = max_length
        self.tokenizer_name = tokenizer_name
        self.tokenizer = tokenizer
        self.add_eos = add_eos
        self.d_output = d_output  # needed for decoder to grab
        self.use_padding = use_padding
        self.upsample = upsample

        # Load sequences
        # Find the transcript name in each header
        pattern = re.compile(r'^EN[A-Z]*ST\d+$')
        self.headers_seqs = [(record.description.split("|"), str(record.seq)) for record in SeqIO.parse(fasta_path, "fasta")]
        self.sequence_ids = [self.get_transcript_id(s[0], pattern) for s in self.headers_seqs]
        
        self.df = pd.read_csv(csv_path)
        mask = self.df['cdna sequences'].apply(lambda x: isinstance(x, str)) & self.df['3UTR sequences'].apply(lambda x: isinstance(x, str))
        self.df = self.df[mask]
        self.df['utr_seqs'] = self.df.apply(lambda row: row['cdna sequences'][-1000:] if (row['3UTR sequences'] == 'Sequenceunavailable') else row['3UTR sequences'], axis=1)

        # Ensure gene names match between the label file and the fasta file
        self.gene_names = list(set(self.df['Transcript ID']) & set(self.sequence_ids))
        self.gene_names.sort()

        if len(self.gene_names) == 0:
            raise ValueError("No matching gene names found between label file and FASTA file.")
        

    def __len__(self):
        return len(self.gene_names)
    
    def get_transcript_id(self, header, pattern):
        matches = filter(pattern.match, header)
        try:
            return next(matches)
        except Exception as e:
            return "None"

    def __getitem__(self, idx):
        gene_name = self.gene_names[idx]
        filtered_df = self.df.loc[self.df['Transcript ID'] == gene_name]

        y = filtered_df.iloc[0][self.synapse_label_name]
        x = filtered_df.iloc[0]['utr_seqs']
        
        seq = self.tokenizer.bos_token + x + self.tokenizer.eos_token
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

        # need to wrap in list
        target = torch.LongTensor([y])  

        return seq, target, {'transcript_id': gene_name}
