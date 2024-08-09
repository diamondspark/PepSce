from modlamp.descriptors import PeptideDescriptor, GlobalDescriptor
import sys
from iFeature.codes.BINARY import *
from iFeature.codes.AAINDEX import *
from iFeature.codes.BLOSUM62 import *
from iFeature.codes.ZSCALE import *
from tqdm import tqdm
import numpy as np

def get_modlamp_descriptors(pep1):
    descriptors= []
#     print (f'peptide is ----> {pep1}')
    try:
        #Global Desc
        desc = GlobalDescriptor(pep1)
        desc.calculate_all(amide= False)
        descriptors.append(desc.descriptor)

        #Peptide Desc
        amp = PeptideDescriptor(pep1,'PPCALI')
        amp.calculate_autocorr(3)
        descriptors.append(amp.descriptor)

        amp = PeptideDescriptor(pep1,'pepcats')
        amp.calculate_crosscorr(3)
        descriptors.append(amp.descriptor)

        amp = PeptideDescriptor(pep1)
        amp.calculate_moment()
        descriptors.append(amp.descriptor)

        amp.calculate_global()
        descriptors.append(amp.descriptor)

        amp = PeptideDescriptor(pep1,'kytedoolittle')
        amp.calculate_profile()
        descriptors.append(amp.descriptor)

        amp = PeptideDescriptor(pep1,'peparc')
        amp.calculate_arc()
        descriptors.append(amp.descriptor)
        return np.concatenate(descriptors,axis=1)
    
    except Exception as e:
#         print(e, pep1, 'modlamp')
        return None
    
def get_ifeat_desc(pep1):
    aa_feature_list = []
    try:
        fasta_str =  [[f'>pep1',f'{pep1}']]
        bin_output = BINARY(fasta_str)
        aai_output = AAINDEX(fasta_str)
        blo_output = BLOSUM62(fasta_str)
        zsl_output = ZSCALE(fasta_str)
        feature_id = bin_output[1][0].split('>')[1]
        bin_output[1].remove(bin_output[1][0])
        aai_output[1].remove(aai_output[1][0])
        blo_output[1].remove(blo_output[1][0])
        zsl_output[1].remove(zsl_output[1][0])
        bin_feature = []
        aai_feature = []
        blo_feature = []
        zsl_feature = []
        for i in range(0, len(bin_output[1]), 20):
            temp = bin_output[1][i:i + 20]
            bin_feature.append(temp)
        for i in range(0, len(aai_output[1]), 531):
            temp = [float(i) for i in aai_output[1][i:i + 531]]
            aai_feature.append(temp)
        for i in range(0, len(blo_output[1]), 20):
            temp = blo_output[1][i:i + 20]
            blo_feature.append(temp)
        for i in range(0, len(zsl_output[1]), 5):
            temp = zsl_output[1][i:i + 5]
            zsl_feature.append(temp)
        aa_fea_matrx = np.hstack([np.array(bin_feature), np.array(aai_feature), np.array(blo_feature), np.array(zsl_feature)])
        return aa_fea_matrx
    except Exception as e:
#         print(e, pep1, 'ifeat')
        return
    
# print(get_ifeat_desc('PYIIKLIYVPKL').shape)