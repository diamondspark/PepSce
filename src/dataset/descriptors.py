from modlamp.descriptors import PeptideDescriptor, GlobalDescriptor
from iFeature.codes.BINARY import *
from iFeature.codes.AAINDEX import *
from iFeature.codes.BLOSUM62 import *
from iFeature.codes.ZSCALE import *
from tqdm import tqdm, tqdm_notebook
import numpy as np

def get_modlamp_descriptors(pep1):
    """
    Return a robust (1,139) descriptor vector.
    If a descriptor fails, the corresponding block is filled with zeros.
    """

    # ---- known descriptor dimensions ----
    # You should confirm these match your modlamp version
    BLOCK_SIZES = [
        10,   # GlobalDescriptor
        57,   # PeptideDescriptor (PPCALI autocorr)
        63,   # PeptideDescriptor (pepcats crosscorr)
        1,    # PeptideDescriptor (moment)
        1,    # PeptideDescriptor (global)
        2,   # PeptideDescriptor (kytedoolittle profile)
        5    # PeptideDescriptor (peparc)
    ]

    TOTAL_DIM = sum(BLOCK_SIZES)  # should be 139
    assert TOTAL_DIM == 139, f"Expected 139 dims, got {TOTAL_DIM}"

    output = np.zeros((1, TOTAL_DIM), dtype=np.float32)

    offset = 0

    def safe_compute(func, size):
        """Run a descriptor safely and return (1,size) array."""
        try:
            desc = func()
            if desc is not None:
                desc = np.asarray(desc)
                if desc.shape[1] == size:
                    return desc
        except Exception as e:
            print(f"Modlamp failed: {e} | peptide: {pep1}")

        return np.zeros((1, size), dtype=np.float32)

    # ---- wrapper lambdas for safe_compute ----
    blocks = []

    # 1. Global Descriptor
    def block_global():
        desc = GlobalDescriptor(pep1)
        desc.calculate_all(amide=False)
        return desc.descriptor
    blocks.append((block_global, BLOCK_SIZES[0]))

    # 2. PPCALI autocorr
    def block_ppcali():
        amp = PeptideDescriptor(pep1, 'PPCALI')
        amp.calculate_autocorr(3)
        return amp.descriptor
    blocks.append((block_ppcali, BLOCK_SIZES[1]))

    # 3. pepcats crosscorr
    def block_pepcats():
        amp = PeptideDescriptor(pep1, 'pepcats')
        amp.calculate_crosscorr(3)
        return amp.descriptor
    blocks.append((block_pepcats, BLOCK_SIZES[2]))

    # 4. moment descriptor
    def block_moment():
        amp = PeptideDescriptor(pep1)
        amp.calculate_moment()
        return amp.descriptor
    blocks.append((block_moment, BLOCK_SIZES[3]))

    # 5. global peptide descriptor
    def block_global2():
        amp = PeptideDescriptor(pep1)
        amp.calculate_global()
        return amp.descriptor
    blocks.append((block_global2, BLOCK_SIZES[4]))

    # 6. kytedoolittle profile
    def block_kd():
        amp = PeptideDescriptor(pep1, 'kytedoolittle')
        amp.calculate_profile()
        return amp.descriptor
    blocks.append((block_kd, BLOCK_SIZES[5]))

    # 7. peparc arc
    def block_peparc():
        amp = PeptideDescriptor(pep1, 'peparc')
        amp.calculate_arc()
        return amp.descriptor
    blocks.append((block_peparc, BLOCK_SIZES[6]))

    # ---- fill output vector ----
    for func, size in blocks:
        desc = safe_compute(func, size)
        output[:, offset:offset+size] = desc
        offset += size

    return output


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
        print(e, pep1, 'ifeat failed')
        return