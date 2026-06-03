# Modified from original source: https://github.com/IBM/DEFT
# Original code licensed under Apache License 2.0

# Changes:
#   - Added CodeCarbon EmissionsTracker for energy tracking
#   - Replaced text/GLUE task with image classification (CIFAR10/CIFAR100)
#   - Replaced AutoAdapterModel/PfeifferConfig with ViTForImageClassification/SeqBnConfig
#   - Replaced AutoTokenizer preprocessing with AutoImageProcessor
#   - Added structured CSV logging
#   - Added per-epoch energy logging (train and validation separately)
#   - Added Python logging (replacing print statements)
#   - Changed optimizer to SGD
#   - Moved output_hook.clear() to start of batch loop

import argparse
import os
import random
import sys
import logging
import csv
import evaluate
import numpy as np
import torch
from datasets import load_dataset
from torch.optim import AdamW, SGD
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    EvalPrediction,
    get_linear_schedule_with_warmup,
    ViTForImageClassification
)
from adapters import init, SeqBnConfig
from codecarbon import EmissionsTracker

#======================================== Setup logger=========================================
# Logger was debugged and written with the help of ChatGPT
logging.getLogger("codecarbon").setLevel(logging.ERROR)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler("run.log"),
        logging.StreamHandler()          # still prints, but only what you log
    ]
)
log = logging.getLogger(__name__)


def log_step(epoch, step, loss, sparsity_loss, density, interval=100):
    if step % interval == 0:
        log.info(
            f"epoch {epoch} | step {step:4d} | "
            f"loss {loss:.4f} | sparsity {sparsity_loss:.4f} | density {density:.4f}"
        )

def kwh_to_j(val):
    KWH_TO_J = 3_600_000
    return round(val * KWH_TO_J, 4) if val != "-" else "-"

def log_epoch(run_id, epoch, eval_metric, mean_density, layerwise, csv_path, epoch_data=None):
    density_pct = mean_density.item() * 100
    layer_str   = ",".join(f"{v.item():.5f}" for v in layerwise)

    log.info(
        f"=== EPOCH {epoch} DONE ===  "
        f"acc={eval_metric["accuracy"]:.4f}  "
        f"density={density_pct:.4f}%  "
        f"layers=[{layer_str}]"
    )

    energy = epoch_data or {}
    

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            run_id, epoch, "-",
            "-", "-", "-",
            eval_metric["accuracy"], round(density_pct, 5), layer_str,
            kwh_to_j(energy.get("energy_consumed", "-")),
            kwh_to_j(energy.get("gpu_energy", "-")),
            kwh_to_j(energy.get("cpu_energy", "-")),
            kwh_to_j(energy.get("ram_energy", "-")),
            energy.get("duration", "-"),
        ])

#============================== Parse arguments etc =======================================================
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def setup_seed(seed):
    # seed = cfg.SEED + utils.get_rank() + 10
    log.info(f"Seed set to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

# Define argparse here
parser = argparse.ArgumentParser()

parser.add_argument("--reduction_factor", type=int, default=8)
parser.add_argument("--non_linearity", type=str, default="relu")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--weight_l", type=float, default=1.0)
parser.add_argument("--model_name_or_path", type=str, default="")
parser.add_argument("--task", type=str, default="cifar10")
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--sparse_obj", action="store_true")
parser.add_argument("--cache_dir", type=str, default="/hf")
parser.add_argument("--max_seq_length", type=int, default=128)
parser.add_argument("--pad_to_max_length", action="store_true")
parser.add_argument("--eps", type=float, default=1e-7)
parser.add_argument("--act_layers_to_save", type=str, default="non_linearity.f")  #"intermediate_act_fn") # change to  "non_linearity"
parser.add_argument("--logname", type=str)
args = parser.parse_args()

log.info(args)

logname=args.logname
# CSV for structured results
CSV_PATH = f"{logname}.csv"
#if not os.path.exists(CSV_PATH):
with open(CSV_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "run_id", "epoch", "step",
        "task_loss", "sparsity_loss", "density",
        "eval_accuracy", "eval_density_pct",
        "layerwise_density", "energy_consumed", 
        "ram_energy","gpu_energy", "cpu_energy", "duration"
    ])


batch_size = args.batch_size
model_name_or_path =args.model_name_or_path 
task = args.task

device = "cuda"
num_epochs = args.epochs
SEED = args.seed
weight_l = args.weight_l
lr = args.lr

sparse_obj = args.sparse_obj



setup_seed(SEED)
is_regression = False
#================================= Load dataset =========================================

if args.task is not None:
    # Downloading and loading a dataset from the hub.
    # Change this to take CIFAR10
    raw_datasets = load_dataset(
        args.task, #    "CIFAR10",
        cache_dir=args.cache_dir,
        use_auth_token= None,
    )


label_list = raw_datasets["train"].features["label"].names
num_labels = len(label_list)

#============================== Image preprocessing =============================================
# AutoImageProcessor was implemented with the help of ChatGPT
feature_extractor = AutoImageProcessor.from_pretrained(
      args.model_name_or_path, cache_dir=args.cache_dir
    )

\
# CIFAR-10 uses "img", CIFAR-100 uses "img" as well
IMG_KEY = "img"

def preprocess(examples):
    # This function was written with the help of ChatGPT
    inputs = feature_extractor(examples[IMG_KEY], return_tensors="pt")
    examples["pixel_values"] = inputs["pixel_values"]
    examples["labels"] = examples["label"]
    return examples

# For sanity check 
raw_datasets["train"] = raw_datasets["train"].select(range(5000))
raw_datasets["test"]  = raw_datasets["test"].select(range(1000))

train_dataset = raw_datasets["train"].with_transform(preprocess)
eval_dataset  = raw_datasets["test"].with_transform(preprocess)


def collate_fn(batch):
    # This function was written with the help of ChatGPT

    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "labels": torch.tensor([x["labels"] for x in batch]),
    }

train_dataloader = DataLoader(
    train_dataset, shuffle=True, collate_fn=collate_fn, batch_size=args.batch_size
)
eval_dataloader = DataLoader(
    eval_dataset, shuffle=False, collate_fn=collate_fn, batch_size=args.batch_size
)


#============================== Load model =============================================

config = AutoConfig.from_pretrained(args.model_name_or_path,num_labels=num_labels, finetuning_task=args.task)
log.info(f"Model config: {config}")

model = ViTForImageClassification.from_pretrained(model_name_or_path,num_labels=num_labels, ignore_mismatched_sizes = True, cache_dir =args.cache_dir)

init(model)


# Add SeqBnConfig
# Implementation and use of SeqBnConfig was written with the help of ChatGPT
adapter_config = SeqBnConfig(
    non_linearity=args.non_linearity,
    reduction_factor=args.reduction_factor,
)
model.add_adapter(args.task, config = adapter_config)


# Activate the adapter
model.train_adapter(args.task)

# Sync label mappings
model.config.label2id = {l: i for i, l in enumerate(label_list)}
model.config.id2label = {i: l for i, l in enumerate(label_list)}



class OutputHook(list):

    """ Hook to capture module outputs.

    """

    def __call__(self, module, input, output):

        self.append(output)

output_hook = OutputHook()


act_layers_to_save =  "non_linearity.f" #"non_linearity" #args.act_layers_to_save 

hooked_layers = []
for name, module in model.named_modules():
    if act_layers_to_save in name:
        hook = module.register_forward_hook(
            output_hook
        )
        hooked_layers.append(name)

log.info(f"Hooked {len(hooked_layers)} layers:")



def compute_l0_penalty(output_hook,eps=1e-7):
    not_zeros = []
    for output in output_hook:
        n_zeros = output**2/(output**2 + eps)
        n_zeros = n_zeros.mean(dim=[1,2])
        n_zeros = n_zeros.reshape(output.shape[0],1)
        not_zeros.append(n_zeros)

    non_zeros = torch.cat(not_zeros,dim=1)

    return non_zeros
    
def compute_eval_sparsity(output_hook):
    not_zeros = []
    for output in output_hook:
        n_zeros = torch.count_nonzero(output,dim=[1,2])/(output.shape[1]*output.shape[2])
        n_zeros = n_zeros.reshape(output.shape[0],1)
        not_zeros.append(n_zeros)

    non_zeros = torch.cat(not_zeros,dim=1)

    return non_zeros


# Get the metric function
metric = evaluate.load("accuracy",experiment_id= f"{args.task}_{args.seed}__{args.sparse_obj}_l0_adapter")

# You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
# predictions and label_ids field) and has to return a dictionary string to float.
def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)
    result = metric.compute(predictions=preds, references=p.label_ids)
    if len(result) > 1:
        result["combined_score"] = np.mean(list(result.values())).item()
    return result


optimizer = AdamW(params=model.parameters(), lr=lr)
optimizer = SGD(params=model.parameters(), lr=lr, momentum=0.9)

# Instantiate scheduler
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0.06 * (len(train_dataloader) * num_epochs),
    num_training_steps=(len(train_dataloader) * num_epochs),
)

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
log.info(f"Trainable Parameters: {trainable_params}")
# also print % 
total_params = sum(p.numel() for p in model.parameters())
log.info(f"Total Parameters: {total_params}")
log.info(f"Trainable %: {trainable_params/total_params*100:.2f}")

acc_list = []
best_acc = 0.0

model.to(device)
all_dense = []

#--- Log run-----------------
run_id = (
    f"{args.task}_{args.model_name_or_path.split("/")[-1]}"
    f"_bs{args.batch_size}_lr{args.lr}"
    f"_rf{args.reduction_factor}_wl{args.weight_l}"
    f"_s{args.seed}"
)
log.info(f"Starting run: {run_id}")
log.info(f"Trainable: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.2f}%)")


with EmissionsTracker(project_name=f"project_name", experiment_id=logname, output_file="output_file.csv", measure_power_secs=1, tracking_mode="process", log_level="error") as tracker:
    epoch_emissions_log_train = []
    epoch_emissions_log_val = []
    for epoch in range(num_epochs):
        model.train()
        tracker.start_task(f"epoch_{epoch+1}_train") 
        for step, batch in enumerate(tqdm(train_dataloader, disable=True)):
            # put every item in batch dictionary to device
        
            # batch = {k: v.to(device)  for k, v in batch.items() if k!="idx"}
            
            batch = {k: v.to(device)  for k, v in batch.items()}
            # print(batch)

            output_hook.clear() # moved here
            outputs = model(**batch)
            loss = outputs.loss

            if sparse_obj:
                not_zeros = compute_l0_penalty(output_hook,eps=args.eps)
                sparsity_loss = not_zeros.mean()#.sum()
                if step % 50 == 0:
                    log_step(epoch, step, loss.item(), sparsity_loss.item(), not_zeros.mean().item())
                # print(sparsity_loss)
                total_loss = weight_l*sparsity_loss + loss
            else:
                total_loss = loss
            #output_hook.clear() not safe

            total_loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        epoch_emissions = tracker.stop_task()
        print(f"Epoch {epoch+1} energy_consumed: {epoch_emissions.energy_consumed}")
        print(f"Epoch {epoch+1} ram_energy: {epoch_emissions.ram_energy}")
        print(f"Epoch {epoch+1} gpu_energy: {epoch_emissions.gpu_energy}")
        print(f"Epoch {epoch+1} cpu_energy: {epoch_emissions.cpu_energy}")
        print(f"Epoch {epoch+1} duration: {epoch_emissions.duration}")
        epoch_data_train = {
            "epoch": epoch + 1,
            "energy_consumed":    epoch_emissions.energy_consumed,
            "ram_energy": epoch_emissions.ram_energy,  # kWh
            "gpu_energy":        epoch_emissions.gpu_energy,
            "cpu_energy":       epoch_emissions.cpu_energy,
            "duration":       epoch_emissions.duration,
        }
        epoch_emissions_log_train.append(epoch_data_train)
        model.eval()
        all_batches_zero = []
        tracker.start_task(f"epoch_{epoch+1}_val") 
        for step, batch in enumerate(tqdm(eval_dataloader, leave=False)):
            batch = {k: v.to(device) for k, v in batch.items()}
            # batch = {k: v.to(device)  for k, v in batch.items() if k!="idx"}
            output_hook.clear()
            with torch.no_grad():
                outputs = model(**batch)
                not_zeros_Eval = compute_eval_sparsity(output_hook)
        
            all_batches_zero.append(not_zeros_Eval)
            

            predictions = outputs.logits #.squeeze()
            predictions, references = predictions, batch["labels"]
            metric.add_batch(
                predictions=predictions.argmax(dim=-1), #predictions=predictions,
                references=references,
            )

        all_batches_zero = torch.cat(all_batches_zero,dim=0)
    
        mean_outs = all_batches_zero.mean(dim=0)
        epoch_emissions = tracker.stop_task()
        epoch_data_val = {
            "epoch": epoch + 1,
            "energy_consumed":    epoch_emissions.energy_consumed,
            "ram_energy": epoch_emissions.ram_energy,  # kWh
            "gpu_energy":        epoch_emissions.gpu_energy,
            "cpu_energy":       epoch_emissions.cpu_energy,
            "duration":       epoch_emissions.duration,
        }
        epoch_emissions_log_val.append(epoch_data_val)

   
        all_dense.append(mean_outs.mean()*100)

        eval_metric = metric.compute()
        acc_list.append(eval_metric["accuracy"]) # to track best model
        if eval_metric["accuracy"] > best_acc:
            best_acc = eval_metric["accuracy"]

        log_epoch(run_id, epoch, eval_metric, mean_outs.mean(), mean_outs, CSV_PATH, epoch_data_val)
  
# if SAVE_WEIGHTS == True:
torch.save(model.state_dict(), f"weights_exp_{run_id}.pth")
model.save_adapter(f"adapter_{run_id}", args.task)

# If needed
#for h in hook_handles: h.remove()
        
log.info(f"all Density: {all_dense}")
log.info(f"acc list: {acc_list}")

all_dense_vals = [f"{d.item():.4f}" for d in all_dense]
log.info(f"All density (%): {all_dense_vals}")
log.info(f"Final density:   {all_dense[-1].item():.4f}%")
log.info(f"Density change:  {((all_dense[0] - all_dense[-1]) / all_dense[0] * 100).item():.2f}% reduction")
log.info(f"Acc list:        {[round(a, 4) for a in acc_list]}")
log.info(f"Best acc:        {best_acc:.4f}")

