import torch
import pandas as pd
from Bio import SeqIO
from pathlib import Path
import math
import re

class SynapseClassificationDataset(torch.utils.data.Dataset):

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
        self.sequences = [x[1] for x in self.headers_seqs]
        self.sequences = dict(zip(self.sequence_ids, self.sequences))

        # Load gene labels
        self.label_df = pd.read_csv(csv_path)

        # Ensure gene names match between the label file and the fasta file
        self.gene_names = list(set(self.label_df['Transcript ID']) & set(self.sequences.keys()))
        self.gene_names.sort()

        if len(self.gene_names) == 0:
            raise ValueError("No matching gene names found between label file and FASTA file.")
        
        # Filter the label DataFrame to keep only the relevant gene names
        self.label_df = self.label_df[self.label_df['Transcript ID'].isin(self.gene_names)]

        if self.upsample:
            # Calculate the oversampling factor k
            class_0_count = (self.label_df[self.synapse_label_name] == 0).sum()
            class_1_count = (self.label_df[self.synapse_label_name] == 1).sum()
            self.k = math.ceil(class_0_count / class_1_count)

            # Oversample class 1 gene names
            class_1_gene_names = self.label_df[self.label_df[self.synapse_label_name] == 1]['Transcript ID'].tolist()
            self.gene_names += class_1_gene_names * (self.k - 1)


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
        x = self.sequences[gene_name]  # Get sequence
        y = self.label_df.loc[self.label_df['Transcript ID'] == gene_name, self.synapse_label_name].values[0]  # Get label
        
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
