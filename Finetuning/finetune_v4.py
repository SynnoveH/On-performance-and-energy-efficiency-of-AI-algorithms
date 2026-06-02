
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import sys, os
import json
import random
import numpy as np
import math
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix,top_k_accuracy_score
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from torchvision import transforms
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from codecarbon import EmissionsTracker, track_emissions
from sklearn.metrics import classification_report
import argparse
import torch.optim as optim
from torch.utils.data import random_split
from torchvision.datasets import CIFAR100
from torch.amp import GradScaler, autocast
from lion_pytorch import Lion
import timm
import os
import logging
print("HF_HOME:", os.environ.get("HF_HOME"))
from codecarbon import EmissionsTracker



# =================== PARSE ARGUMENTS=========================================
# Explanation: used args to create experiment files
parser = argparse.ArgumentParser()
parser.add_argument("--exp_id", type=int, required=True)
parser.add_argument("--save_emission_to", type=str, default="emissions.csv")
parser.add_argument("--model_id", type=str, default="vit_b_32")
parser.add_argument("--epochs", type=int, default=40)
parser.add_argument("--lr", type=float, default=0.00010)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--optimizer", type=str, default="Adam")
parser.add_argument("--project_name", type=str, default="untitled")
parser.add_argument("--dropout", type=float, default=0.1)   
parser.add_argument("--weight_decay", type=float, default=0.1)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()


exp_id = args.exp_id
save_to = args.save_emission_to
model_id = args.model_id
epochs = args.epochs
lr = args.lr
batch_size = args.batch_size
optimizer_name = args.optimizer
dropout = args.dropout
weight_decay = args.weight_decay
project_seed = args.seed
project_name = args.project_name
global_save_to = Path(project_name)

print("Args:", args)

SAVE_WEIGHTS = False

#==========================Logger ===============
logger = logging.getLogger("experiment_logger")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
formatter = logging.Formatter(
    "[%(name)s %(levelname)s @ %(asctime)s] %(message)s",
    datefmt="%H:%M:%S"
)
handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(handler)

#==================================== SEED ==========================
# Created seeds for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ================================== MODEL ========================================
# Function to load models from timm
# Used for ViT-S/32, ViT-S/16, ViT-Ti/16
def load_timm_model(pretrained=True, model_id = model_id, dropout=0.0):
        model = timm.create_model(model_id, pretrained=pretrained, num_classes=100, drop_rate=dropout)
        return model


# ================================== Optimizer ========================================
# Function to get optimizers. Experimented with different optimizers in early stage but ended up using SGD
def get_optimizer(optimizer_name):
    if optimizer_name == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    
    elif optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        
    elif optimizer_name == "Lion":
        optimizer = Lion(model.parameters(), lr=lr, weight_decay=weight_decay)
        
    elif optimizer_name == "Adafactor":
        optimizer = torch.optim.Adafactor(model.parameters(), lr=lr, weight_decay=weight_decay)

    else:
        raise ValueError(f"Unknown optimizer {optimizer_name}")
    
    return optimizer

# ============================== COSINE SCHEDULE WITH WARMUP ============================
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
    # Schedule the learning rate with a linear warmup with values between 0 and initial learning rate.
    # After num_training_steps the values decrease between initial lr and 0, as it follow the cosine function 
    # This section (lr_lambda) was debugged and written with the help of ChatGPT
    
    def lr_lambda(current_step):
   
        # Warmup phase
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        # Cosine decay phase
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress)))
    
    return LambdaLR(optimizer, lr_lambda, last_epoch)

# =============================== LOAD MODEL ============================================
# Load function for model
def load_vit(model_name: str, device=device, dropout=0.0):
    if model_name not in vit_models:
        raise ValueError(f"Unknown model {model_name}. Available: {list(vit_models.keys())}")

    model_fn, weights = vit_models[model_name]

    torchvision_models =  ["vit_b_16", "vit_b_32","vit_l_32"]
    
    if model_name in torchvision_models:
        print("torchivision model")
        # torchvision models
        model = model_fn(weights=weights, dropout=dropout)
        model.heads[0] = torch.nn.Linear(model.heads[0].in_features, 100)

    else:
        print("timm model")
        model = model_fn(dropout=dropout)
    
    return model.to(device)


# ============================== TRAIN MODEL ============================
# Function to train model
# Uses Gradient scalar  to protect small gradients, clipping to protect against exploding gradients
# Use mixed precision to speed up training
# Run validation on every epoch
# Calculate accuracy, recall, specificity, precision, F1-score

def train_model(loaded_model, epochs, lr, batch_size, train_loader, val_loader, dropout, weight_decay, scheduler, optimizer, tracker):
    print(f"Starting training for model with set up: | epochs {epochs} | lr {lr} | batch size {batch_size} | optimizer {optimizer_name} | dropout {dropout} | weight decay {weight_decay} | scheduler cosine_schedule_with_warmup |  seed {project_seed}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("CUDA available:", torch.cuda.is_available())

    model = loaded_model
    # Initialize Gradient Scaler
    scaler = GradScaler()
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    loss_list = []
    accuracy_list = []
    precision_list = []

    val_loss_list = []
    val_accuracy_list = []
    val_precision_list = []
    val_recall_list = []
    val_f1_list = []
    val_specificity_list = []

    global_step = 0
    epoch_emissions_log = []
    logger.info(f"FINISHED LOADING")
    for epoch in range(epochs):
        logger.info(f"EPOCH_START {epoch+1}")
        tracker.start_task(f"epoch_{epoch+1}") 
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            #Use Mixed precision to speed up training
            with autocast(device_type="cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            # Scale loss to protect small gradients
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)  # Unscale gradients before clipping when using GradScaler
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Protect against exploding gradients
            
            # Optimizer step
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Update scheduler
            global_step += 1 
            scheduler.step()

        avg_loss = running_loss / len(train_loader)
        accuracy = correct / total
        precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)

        loss_list.append(avg_loss)
        accuracy_list.append(accuracy)
        precision_list.append(precision)


# ===================================== Validation ====================================================
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        val_preds = []
        val_labels = []
        val_probs = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                with autocast(device_type="cuda"):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                val_running_loss += loss.item()

                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

                # Get predicted probabilities (softmax)
                probs = torch.softmax(outputs, dim=1)
                val_probs.extend(probs.cpu().numpy())


#==================================== Performance logging ===================================================

        print(f"Epoch {epoch+1}, LR: {scheduler.get_last_lr()[0]:.6f}")
        epoch_emissions = tracker.stop_task()
        print(f"Epoch {epoch+1} energy_consumed: {epoch_emissions.energy_consumed}")
        print(f"Epoch {epoch+1} ram_energy: {epoch_emissions.ram_energy}")
        print(f"Epoch {epoch+1} gpu_energy: {epoch_emissions.gpu_energy}")
        print(f"Epoch {epoch+1} cpu_energy: {epoch_emissions.cpu_energy}")
        print(f"Epoch {epoch+1} duration: {epoch_emissions.duration}")
        epoch_data = {
            "epoch": epoch + 1,
            "energy_consumed":    epoch_emissions.energy_consumed,
            "ram_energy": epoch_emissions.ram_energy,  # kWh
            "gpu_energy":        epoch_emissions.gpu_energy,
            "cpu_energy":       epoch_emissions.cpu_energy,
            "duration":       epoch_emissions.duration,
        }
        epoch_emissions_log.append(epoch_data)
        """
        Performance logging
        """
        # average loss and accuracy
        val_avg_loss = val_running_loss / len(val_loader)
        val_accuracy = val_correct / val_total

        # precison, recall, F1
        val_precision = precision_score(val_labels, val_preds, average="macro", zero_division=0)
        val_recall = recall_score(val_labels, val_preds, average="macro", zero_division=0)
        val_f1 = f1_score(val_labels, val_preds, average="macro", zero_division=0)
        
        cm = confusion_matrix(val_labels, val_preds)

        # Calculate specificity
        # This section (calculate specificity) was debugged written with the help of ChatGPT
        tn = cm.sum() - cm.sum(axis=1)[:, None] - cm.sum(axis=0)[None, :] + np.diag(cm)
        fp = cm.sum(axis=0) - np.diag(cm)
        specificity_per_class = tn / (tn + fp)
        val_specificity = specificity_per_class.mean()


        val_loss_list.append(val_avg_loss)
        val_accuracy_list.append(val_accuracy)
        val_precision_list.append(val_precision)
        val_recall_list.append(val_recall)
        val_f1_list.append(val_f1)
        val_specificity_list.append(val_specificity)
        print(f"Global steps {global_step}")
        print(f"Epoch [{epoch+1}/{epochs}] Training Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        print(f"Epoch [{epoch+1}/{epochs}] Validation Loss: {val_avg_loss:.4f}, Accuracy: {val_accuracy:.4f}")

        logger.info(
        f"Epoch [{epoch+1}/{epochs}] |"
        f"Glob step: {global_step} |"
        f"Train Loss: {avg_loss:.4f}, Acc: {accuracy:.4f} | "
        f"Val Loss: {val_avg_loss:.4f}, Acc: {val_accuracy:.4f}"
    )
        logger.info(f"EPOCH_END {epoch+1}")
    
    
    cm = confusion_matrix(val_labels, val_preds)

    return model, avg_loss, accuracy, loss_list, accuracy_list, precision_list, val_loss_list, val_accuracy_list, val_precision_list, cm, val_recall_list, val_specificity_list, val_f1_list, epoch_emissions_log


# ============================== TEST MODEL ============================
def run_inference(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            probs = torch.softmax(outputs, dim=1)
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Accuracy, Precision, Recall, F1
    accuracy = np.mean(all_preds == all_labels)
    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall = recall_score(all_labels, all_preds, average = "macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average = "macro", zero_division=0)

    cm = confusion_matrix(all_labels, all_preds)
    # Calculate specificity
    # This section (calculate specificity) was debugged written with the help of ChatGPT

    tn = cm.sum() - cm.sum(axis=1)[:, None] - cm.sum(axis=0)[None, :] + np.diag(cm)
    fp = cm.sum(axis=0) - np.diag(cm)
    specificity_per_class = tn / (tn + fp)
    specificity = specificity_per_class.mean()
    #specificity_per_class = specificity_per_class.tolist()

    #Top k accuracy
    topk_values=[1, 5]
    topk_results = {}
    for k in topk_values:
        topk_results[f"top{k}_accuracy"] = top_k_accuracy_score(all_labels, all_probs, k=k)

    # Classification report
    clf_report = classification_report(all_labels, all_preds)

    cm = confusion_matrix(all_labels, all_preds)

    results = {
        "accuracy": accuracy,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "specificity_macro": specificity,
        "confusion_matrix": cm.tolist(),
        "classification_report": clf_report,
        **topk_results
    }

    save_test_results_to = global_save_to / "test_results" / f"test_results_exp_{exp_id}.json"
    with open(save_test_results_to, "w") as f:
        json.dump(results, f, indent=4)


    print(f"Test results saved for experiment {exp_id}")
    return accuracy, precision, recall, specificity 


#===================================CIFAR100 TRANSFORM =============================
train_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.5071, 0.4867, 0.4408],
                         [0.2675, 0.2565, 0.2761])
])

val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.5071, 0.4867, 0.4408],
                         [0.2675, 0.2565, 0.2761])
])



# ===================== HANDLING MODELS ============================================
selected_model = model_id
if selected_model == "vit_b_16":
    from torchvision.models import vit_b_16, ViT_B_16_Weights
    vit_models =  {"vit_b_16": (vit_b_16, ViT_B_16_Weights.DEFAULT)}
elif selected_model == "vit_b_32":
    from torchvision.models import vit_b_32, ViT_B_32_Weights
    vit_models = {"vit_b_32": (vit_b_32, ViT_B_32_Weights.DEFAULT)}

elif selected_model == "vit_l_32":
    from torchvision.models import vit_l_32, ViT_L_32_Weights
    vit_models = {"vit_l_32": (vit_l_32, ViT_L_32_Weights.DEFAULT)}

elif selected_model == "vit_small_patch32_224":
    # timm model
    vit_models = {"vit_small_patch32_224": (load_timm_model, None)}

elif selected_model == "vit_small_patch16_224":
    # timm model
    vit_models = {"vit_small_patch16_224": (load_timm_model, None)}

elif selected_model == "vit_tiny_patch16_224":
    # timm model
    vit_models = {"vit_tiny_patch16_224": (load_timm_model, None)}
else:
    raise ValueError(f"Unknown model {selected_model}")

set_seed(project_seed)

#================================== Data ===================================
# Use the same transform but resize to 224x224 (which is the ViT input size for the models implemented)
train_transforms = train_transforms

val_transforms = val_transforms

# Load the full CIFAR100 training dataset
full_train_dataset = CIFAR100(root="./data", train=True, download=True, transform=train_transforms)

# Split the training data into train and validation sets (80/20 split)
train_size = int(0.8 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size

generator = torch.Generator().manual_seed(project_seed)
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size], generator=generator)

# Assign transformation
val_dataset.dataset.transform = val_transforms

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(project_seed))
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


#============================= Scheduler and optimizer ============================
#Calculate total steps
num_training_steps = epochs*len(train_loader)
num_warmup_steps = 500 

optimizer = get_optimizer(optimizer_name)

scheduler = get_cosine_schedule_with_warmup(
    optimizer, 
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)


#================================== Finetune===================================
print("="*10)
print(f"Running experiment {exp_id}")
print(f"Model:  {model_id}, batch_size: {batch_size}, learning_rate {lr} testing {optimizer_name}")
print("="*10)
project_name_training =project_name + "finetune.csv"

loaded_model = load_vit(model_id, dropout=dropout)
save_to_training = project_name_training


#========================= Fine tune ================================
with EmissionsTracker(project_name=f"{model_id}_training", experiment_id=exp_id, output_file=save_to_training, measure_power_secs=10, tracking_mode="process") as tracker:
    model, avg_loss, accuracy, loss_list, accuracy_list, precision_list, val_loss_list, val_accuracy_list, val_precision_list, val_cm, val_recall_list, val_specificity_list, val_f1_list, epoch_emissions_log = train_model(
        loaded_model=loaded_model, epochs=epochs, lr=lr, batch_size=batch_size, train_loader=train_loader, val_loader=val_loader, dropout=dropout, weight_decay=weight_decay, scheduler = scheduler, optimizer = optimizer, tracker=
        tracker)

# Save weights and results...
if SAVE_WEIGHTS == True:
    torch.save(model.state_dict(), f"{model_id}_weights_exp{exp_id}.pth")

#================================== Test ===================================
#====================================== Data Transforms and Data Loader ===========================================
test_transforms = val_transforms

#====================================== Load Test Dataset  ===========================================
test_dataset = CIFAR100(root="./data", train=False, download=True, transform=test_transforms)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
print("Running test after training")


project_name_training = project_name + "inference.csv"
save_to_inference = project_name_training

#========================================== Test =============================================================
with EmissionsTracker(project_name=f"{model_id}_inference", experiment_id=exp_id, output_file=save_to_inference, measure_power_secs=10, tracking_mode="process") as tracker:
    test_accuracy, test_precision, test_recall, test_specificity = run_inference(model, test_loader)


#========================================== Save results =============================================================
results = {
    "model_id": model_id,
    "exp_id": exp_id,
    "avg_loss": avg_loss,
    "accuracy": accuracy,
    "loss_list": loss_list,
    "accuracy_list": accuracy_list,
    "precision_list": precision_list,
    "val_loss_list": val_loss_list,
    "val_accuracy_list": val_accuracy_list,
    "val_precision_list": val_precision_list,
    "val_recall_list": val_recall_list,
    "val_specificity_list": val_specificity_list, 
    "val_f1_list": val_f1_list,
    "batch_size": batch_size,
    "lr": lr,
    "dropout": dropout,
    "weight_decay": weight_decay,
    "epochs": epochs,
    "optimizer": optimizer_name,
    "test_accuracy":test_accuracy,
    "test_precision": test_precision,
    "test_recall": test_recall,
    "test_specificity": test_specificity,
    "epoch_emissions_log": epoch_emissions_log
}

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return super(NumpyEncoder, self).default(obj)

save_performance_to = global_save_to / "performance" / f"{model_id}_performance_exp{exp_id}.json"
with open(save_performance_to, "w") as f:
    json.dump(results, f, indent=4, cls=NumpyEncoder)

save_confusion_matrix = global_save_to / "val_cm" / f"val_confusion_matrix_exp{exp_id}.npy"
np.save(save_confusion_matrix, np.array(val_cm))


print("="*10)
print(f"Finished experiment {exp_id}")
print("="*10)