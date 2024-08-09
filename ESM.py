import esm
import torch
from tqdm import tqdm


class ESMEmbed():
    def __init__(self, device) -> None:
        self.device = device #torch.device('cuda:2')# device #
        self.esm_model, alphabet = esm.pretrained.esm1_t6_43M_UR50S()
        self.esm_model = self.esm_model.to(self.device)
        self.esm_model.eval()
        self.batch_converter = alphabet.get_batch_converter()
        

    def get_esm(self, data,long_seq, representation_layer):
        '''data: protein seq: str
        long_seq: Boolean >1022
        '''
        if long_seq:
            b = [data[i:i+1022] for i in range(len(data)-1021)]  #sliding window to get 1022 amino acids sequences
            data = [subprot for subprot in enumerate(b)]  
            chunks = [data[x:x+1] for x in range(0, len(data), 1)]  #chunks is creating batches of size 50
            sequence_representations = []
            for data in chunks:
                batch_labels, batch_strs, batch_tokens = batch_converter(data)

                with torch.no_grad():
                    results = esm_model(batch_tokens.to(device), repr_layers=[representation_layer], return_contacts=True)
                    token_representations = results["representations"][representation_layer].cpu().detach()
                for i, (_, seq) in enumerate(data):
                    sequence_representations.extend([token_representations[i, 1 : len(seq) + 1].mean(0)])
        else:
            data = [(0,data)]
            batch_labels, batch_strs, batch_tokens = self.batch_converter(data)

            with torch.no_grad():
                results = self.esm_model(batch_tokens.to(self.device), repr_layers=[representation_layer], return_contacts=True)
                token_representations = results["representations"][representation_layer]

            sequence_representations = []
            for i, (_, seq) in enumerate(data):
                sequence_representations.append(token_representations[i, 1 : len(seq) + 1].mean(0))
                
        mean = torch.mean(torch.stack(sequence_representations),dim = 0)
        return mean

# get_esm('PAVEAQIEKLLA',False,6).shape


# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument("--device", help="CPU/CUDA")
# args = parser.parse_args()

# esm_helper = ESMEmbed(args) 
# print(esm_helper.get_esm('PAVEAQIEKLLA',False,6).shape)
# print(args.device)

