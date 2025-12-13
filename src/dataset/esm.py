#ESM
import subprocess
import os
import torch
from pathlib import Path
from tqdm import tqdm
from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained, MSATransformer

# def run(args):
#     model, alphabet = pretrained.load_model_and_alphabet(args.model_location)
#     model.eval()
#     if isinstance(model, MSATransformer):
#         raise ValueError(
#             "This script currently does not handle models with MSA input (MSA Transformer)."
#         )
#     if torch.cuda.is_available():
#         model = model.cuda()
#         print("Transferred model to GPU")

#     dataset = FastaBatchedDataset.from_file(args.fasta_file)
#     batches = dataset.get_batch_indices(args.toks_per_batch, extra_toks_per_seq=1)
#     data_loader = torch.utils.data.DataLoader(
#         dataset, collate_fn=alphabet.get_batch_converter(args.truncation_seq_length), batch_sampler=batches
#     )
#     print(f"Read {args.fasta_file} with {len(dataset)} sequences")

#     args.output_dir.mkdir(parents=True, exist_ok=True)
#     return_contacts = "contacts" in args.include

#     assert all(-(model.num_layers + 1) <= i <= model.num_layers for i in args.repr_layers)
#     repr_layers = [(i + model.num_layers + 1) % (model.num_layers + 1) for i in args.repr_layers]

#     with torch.no_grad():
#         for batch_idx, (labels, strs, toks) in enumerate(data_loader):
#             print(
#                 f"Processing {batch_idx + 1} of {len(batches)} batches ({toks.size(0)} sequences)"
#             )
#             if torch.cuda.is_available() and not args.nogpu:
#                 toks = toks.to(device="cuda", non_blocking=True)

#             out = model(toks, repr_layers=repr_layers, return_contacts=return_contacts)

#             logits = out["logits"].to(device="cpu")
#             representations = {
#                 layer: t.to(device="cpu") for layer, t in out["representations"].items()
#             }
#             if return_contacts:
#                 contacts = out["contacts"].to(device="cpu")

#             for i, label in enumerate(labels):
#                 args.output_file = args.output_dir / f"{label}.pt"
#                 args.output_file.parent.mkdir(parents=True, exist_ok=True)
#                 result = {"label": label}
#                 truncate_len = min(args.truncation_seq_length, len(strs[i]))
#                 # Call clone on tensors to ensure tensors are not views into a larger representation
#                 # See https://github.com/pytorch/pytorch/issues/1995
#                 if "per_tok" in args.include:
#                     result["representations"] = {
#                         layer: t[i, 1 : truncate_len + 1].clone()
#                         for layer, t in representations.items()
#                     }
#                 if "mean" in args.include:
#                     result["mean_representations"] = {
#                         layer: t[i, 1 : truncate_len + 1].mean(0).clone()
#                         for layer, t in representations.items()
#                     }
#                 if "bos" in args.include:
#                     result["bos_representations"] = {
#                         layer: t[i, 0].clone() for layer, t in representations.items()
#                     }
#                 if return_contacts:
#                     result["contacts"] = contacts[i, : truncate_len, : truncate_len].clone()

#                 torch.save(
#                     result,
#                     args.output_file,
#                 )


def run_esm(
    model_location: str,
    fasta_file: str,
    output_dir: str,
    toks_per_batch: int = 4096,
    repr_layers=[6],
    include=["mean"],
    truncation_seq_length=1022,
    use_gpu=True,
):
    """
    Run ESM embedding extraction directly without argparse.

    Parameters
    ----------
    model_location : str
        Path or pretrained ESM model name (e.g., "esm1_t6_43M_UR50S")
    fasta_file : str
        FASTA file path
    output_dir : str
        Where to store PT files
    toks_per_batch : int
        Max batch size
    repr_layers : list[int]
        Model layers to extract representations from
    include : list[str]
        "mean", "per_tok", "bos", "contacts"
    truncation_seq_length : int
        Maximum sequence length
    use_gpu : bool
        If True and CUDA available, use GPU
    """

    fasta_file = Path(fasta_file)
    output_dir = Path(output_dir)

    model, alphabet = pretrained.load_model_and_alphabet(model_location)
    model.eval()

    if isinstance(model, MSATransformer):
        raise ValueError("MSA Transformer models are not supported here.")

    if torch.cuda.is_available() and use_gpu:
        model = model.cuda()
        print("Transferred model to GPU")

    dataset = FastaBatchedDataset.from_file(fasta_file)
    batches = dataset.get_batch_indices(toks_per_batch, extra_toks_per_seq=1)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=alphabet.get_batch_converter(truncation_seq_length),
        batch_sampler=batches,
    )

    print(f"Read {fasta_file} with {len(dataset)} sequences")
    output_dir.mkdir(parents=True, exist_ok=True)

    return_contacts = "contacts" in include

    # Normalize repr_layers similar to original code
    assert all(-(model.num_layers + 1) <= i <= model.num_layers for i in repr_layers)
    norm_layers = [(i + model.num_layers + 1) % (model.num_layers + 1) for i in repr_layers]

    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in tqdm(enumerate(data_loader)):
            print(f"Processing batch {batch_idx + 1}/{len(batches)} ({toks.size(0)} sequences)")

            if torch.cuda.is_available() and use_gpu:
                toks = toks.to(device="cuda", non_blocking=True)

            out = model(toks, repr_layers=norm_layers, return_contacts=return_contacts)

            representations = {layer: t.to("cpu") for layer, t in out["representations"].items()}
            if return_contacts:
                contacts = out["contacts"].to("cpu")
            
            batch_output = {}

            for i, label in enumerate(labels):
                truncate_len = min(truncation_seq_length, len(strs[i]))
                entry = {"label": label, 
                         "peptide": strs[i]}
                if "mean" in include:
                    entry["mean"] = {
                        layer: t[i, 1:truncate_len+1].mean(0).clone()
                        for layer, t in representations.items()
                    }
                batch_output[label] = entry

            # save per batch
            output_file = output_dir / f"batch_{batch_idx:05d}.pt"
            torch.save(batch_output, output_file)

            

            # # Loop over sequences inside the batch
            # for i, label in enumerate(labels):

            #     result = {"label": label}
            #     truncate_len = min(truncation_seq_length, len(strs[i]))

            #     if "per_tok" in include:
            #         result["representations"] = {
            #             layer: t[i, 1:truncate_len+1].clone()
            #             for layer, t in representations.items()
            #         }

            #     if "mean" in include:
            #         result["mean_representations"] = {
            #             layer: t[i, 1:truncate_len+1].mean(0).clone()
            #             for layer, t in representations.items()
            #         }

            #     if "bos" in include:
            #         result["bos_representations"] = {
            #             layer: t[i, 0].clone()
            #             for layer, t in representations.items()
            #         }

            #     if return_contacts:
            #         result["contacts"] = contacts[i, :truncate_len, :truncate_len].clone()

            #     output_file = output_dir / f"{label}.pt"
            #     torch.save(result, output_file)

    print("Embedding extraction completed.")



def run_esm_extraction(peptides,output_dir):
    """
    Runs ESM embedding extraction using the specified peptides FASTA name.
    """
    fasta_path = f"{peptides}"
    os.makedirs(output_dir, exist_ok=True)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    run_esm(
    model_location = "esm1_t6_43M_UR50S",
    fasta_file = fasta_path,
    output_dir = output_dir,
    toks_per_batch = 4096,
    repr_layers=[6],
    include=["mean"],
    truncation_seq_length=1022,
    use_gpu=True,
)
    
    
    # cmd = [
    #     "python", "./esm/scripts/extract.py",
    #     "esm1_t6_43M_UR50S",
    #     fasta_path,
    #     output_dir,
    #     "--repr_layers", "6",
    #     "--include", "mean"
    # ]

    # print("Running command:")
    # print(" ".join(cmd))

    # # Execute and wait
    # subprocess.run(cmd, check=True)
    # print("ESM extraction completed.")