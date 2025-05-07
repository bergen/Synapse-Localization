import torch
from Bio import SeqIO
import re

class CodonScoring(torch.utils.data.Dataset):

    def __init__(
        self,
        fasta_path,
        max_length,
        d_output=2,  # default binary classification
        tokenizer=None,
        tokenizer_name=None,
        use_padding=True,
        add_eos=False
    ):
        self.max_length = max_length
        self.tokenizer_name = tokenizer_name
        self.tokenizer = tokenizer
        self.add_eos = add_eos
        self.d_output = d_output
        self.use_padding = use_padding

        # Load sequences
        # Find the transcript name in each header
        pattern = re.compile(r'^EN[A-Z]*ST\d+$')
        self.headers_seqs = [(record.description.split("|"), str(record.seq)) for record in SeqIO.parse(fasta_path, "fasta")]
        self.sequence_ids = [self.get_transcript_id(s[0], pattern) for s in self.headers_seqs]
        self.sequences = [x[1] for x in self.headers_seqs]
        self.gene_names = [id for id in self.sequence_ids if id != "None"]

        if len(self.gene_names) == 0:
            raise ValueError("No valid gene names found in the FASTA file.")

    def __len__(self):
        return len(self.gene_names)
    
    def get_transcript_id(self, header, pattern):
        matches = filter(pattern.match, header)
        try:
            return next(matches)
        except StopIteration:
            return "None"

    def __getitem__(self, idx):
        gene_name = self.gene_names[idx]
        x = self.sequences[idx]  # Get sequence
        
        seq = self.tokenizer.bos_token + x + self.tokenizer.eos_token
        seq = self.tokenizer(
            seq,
            add_special_tokens=False,
            padding="max_length" if self.use_padding else 'do_not_pad',
            max_length=self.max_length+2,  # allow for bos and eos tokens
            truncation=True,
        )  
        seq = seq["input_ids"]  # get input_ids

        # convert to tensor
        seq = torch.LongTensor(seq)  

        # Since we don't have labels, we'll return a placeholder target
        target = torch.LongTensor([0])  

        return seq, target, {'transcript_id': gene_name}