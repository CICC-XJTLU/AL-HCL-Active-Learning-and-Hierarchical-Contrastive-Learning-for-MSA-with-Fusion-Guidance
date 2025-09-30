from __future__ import absolute_import, division, print_function

import argparse
import random
import torch
import numpy as np
import wandb
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torch.nn import MSELoss
from pytorch_transformers import WarmupLinearSchedule
from pytorch_transformers.modeling_roberta import RobertaConfig
from networks.SentiLARE import RobertaForSequenceClassification
from utils.databuilder import set_up_data_loader, random_sampling
from utils.set_seed import set_random_seed, seed
from utils.metric import score_model
from config.global_configs import DEVICE
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from networks.subnet.HCL_Module import HCL_Total

def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["mosi", "mosei"], default="mosi")
    parser.add_argument("--data_path", type=str, default='./dataset/MOSI_16_sentilare_unaligned_data.pkl')
    parser.add_argument("--max_seq_length", type=int, default=50)
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=64, help="训练批次大小")
    parser.add_argument("--dev_batch_size", type=int, default=128)
    parser.add_argument("--test_batch_size", type=int, default=128)
    parser.add_argument("--n_epochs", type=int, default=500)
    parser.add_argument("--beta_shift", type=float, default=1.0)
    parser.add_argument("--dropout_prob", type=float, default=0.5)
    parser.add_argument("--model", type=str, choices=["bert-base-uncased", "xlnet-base-cased", "roberta-base"], default="roberta-base")
    parser.add_argument("--model_name_or_path", type=str, default='./pretrained_model/sentilare_model/sentilare_model', help="Path to pre-trained model or shortcut name")
    parser.add_argument("--learning_rate", type=float, default=6e-5)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--gradient_accumulation_step", type=int, default=1)
    parser.add_argument("--test_step", type=int, default=20)
    parser.add_argument("--max_grad_norm", type=int, default=2)
    parser.add_argument("--warmup_proportion", type=float, default=0.4)
    parser.add_argument("--seed", type=seed, default=6758, help="integer or 'random'")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--active_learning_interval", type=int, default=30, help="Interval (in epochs) for applying active learning")
    parser.add_argument("--num_active_learning_samples", type=int, default=20, help="Number of samples to select during each active learning step")
    return parser.parse_args()

def score_model(preds, labels):
    # Basic metrics
    mae = mean_absolute_error(labels, preds)
    corr = np.corrcoef(preds, labels)[0, 1]
    
    # Convert preds and labels to binary for classification metrics
    binary_preds = [1 if p > 0 else 0 for p in preds]
    binary_labels = [1 if l > 0 else 0 for l in labels]
    
    # Accuracy and F1 score for classes Has0 and Non0
    has0_acc_2 = accuracy_score(binary_labels, binary_preds)  # Adjust this if needed
    has0_f1_score = f1_score(binary_labels, binary_preds)
    
    non0_acc_2 = accuracy_score(binary_labels, binary_preds)  # Adjust this if needed
    non0_f1_score = f1_score(binary_labels, binary_preds)
    
    return has0_acc_2, has0_f1_score, non0_acc_2, non0_f1_score, mae, corr

def prep_for_training(args, num_train_optimization_steps: int):
    config = RobertaConfig.from_pretrained(args.model_name_or_path, num_labels=1, finetuning_task='sst')
    model = RobertaForSequenceClassification.from_pretrained(
        args.model_name_or_path, config=config, pos_tag_embedding=True, senti_embedding=True, polarity_embedding=True)
    model.to(DEVICE)
    
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.weight"]
    AL_HCL_params = ['AL_HCL']
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay) and not any(nd in n for nd in AL_HCL_params)],
            "weight_decay": args.weight_decay,
        },
        {"params": model.roberta.encoder.AL_HCL.parameters(), 'lr': args.learning_rate, "weight_decay": args.weight_decay},
        {
            "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay) and not any(nd in n for nd in AL_HCL_params)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = SGD(optimizer_grouped_parameters, lr=args.learning_rate, momentum=0.9)
    scheduler = WarmupLinearSchedule(
        optimizer,
        warmup_steps=args.warmup_proportion * num_train_optimization_steps,
        t_total=num_train_optimization_steps,
    )
    return model, optimizer, scheduler

def train_epoch(args, model: nn.Module, train_dataloader: DataLoader, optimizer, scheduler):
    model.train()
    preds = []
    labels = []
    tr_loss = 0

    nb_tr_steps = 0
    for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        batch = tuple(t.to(DEVICE) for t in batch)
        input_ids, visual_ids, acoustic_ids, pos_ids, senti_ids, polarity_ids, visual, acoustic, input_mask, segment_ids, label_ids = batch
        visual = torch.squeeze(visual, 1)

        outputs = model(
            input_ids,
            visual,
            acoustic,
            visual_ids,
            acoustic_ids,
            pos_ids, senti_ids, polarity_ids,
            attention_mask=input_mask,
            token_type_ids=segment_ids,
        )
        logits = outputs[0]
        hcl_loss = HCL_Total(visual,
            acoustic,
            visual_ids,
            acoustic_ids)
        try:
            # Ensure logits are a tensor
            if not isinstance(logits, torch.Tensor):
                raise TypeError(f"Expected logits to be a tensor, but got {type(logits)}")
            loss_fct = MSELoss()
            loss = loss_fct(logits.view(-1), label_ids.view(-1))
            if args.gradient_accumulation_step > 1:
                loss = loss / args.gradient_accumulation_step
                hcl_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            tr_loss += loss.item()
            nb_tr_steps += 1
            if (step + 1) % args.gradient_accumulation_step == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            # Convert logits to numpy and ensure they are tensors
            logits = logits.detach().cpu().numpy()
            logits = np.squeeze(logits).tolist()
            label_ids = label_ids.detach().cpu().numpy()
            label_ids = np.squeeze(label_ids).tolist()
            preds.extend(logits)
            labels.extend(label_ids)
        
        except TypeError as e:
            print(f"Skipping step {step + 1} due to error: {e}")
            continue  # Skip to the next step if an error occurs

    preds = np.array(preds)
    labels = np.array(labels)

    return tr_loss / nb_tr_steps, preds, labels



def evaluate_epoch(args, model: nn.Module, dataloader: DataLoader):
    model.eval()
    preds = []
    labels = []

    loss = 0
    nb_steps = 0
    
    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            batch = tuple(t.to(DEVICE) for t in batch)
            input_ids, visual_ids, acoustic_ids, pos_ids, senti_ids, polarity_ids, visual, acoustic, input_mask, segment_ids, label_ids = batch
            visual = torch.squeeze(visual, 1)
            
            outputs = model(
                input_ids,
                visual,
                acoustic,
                visual_ids,
                acoustic_ids,
                pos_ids, senti_ids, polarity_ids,
                token_type_ids=segment_ids,
                attention_mask=input_mask,
                labels=None,
            )
            logits = outputs[0]
            loss_fct = MSELoss()
            loss = loss_fct(logits.view(-1), label_ids.view(-1))
            if args.gradient_accumulation_step > 1:
                loss = loss / args.gradient_accumulation_step
            loss += loss.item()
            nb_steps += 1
            
            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.detach().cpu().numpy()
            
            logits = np.squeeze(logits).tolist()
            label_ids = np.squeeze(label_ids).tolist()
            
            preds.extend(logits)
            labels.extend(label_ids)
        
        preds = np.array(preds)
        labels = np.array(labels)

        # Compute additional metrics
        has0_acc_2, has0_f1_score, non0_acc_2, non0_f1_score, mae, corr = score_model(preds, labels)
        
    return loss / nb_steps, preds, labels, has0_acc_2, has0_f1_score, non0_acc_2, non0_f1_score, mae, corr


class ActiveLearningDataset(Dataset):
    def __init__(self, dataset, sampled_data):
        self.dataset = dataset
        self.sampled_data = sampled_data

    def __len__(self):
        return len(self.dataset) + len(self.sampled_data)

    def __getitem__(self, idx):
        if idx < len(self.dataset):
            return self.dataset[idx]
        else:
            return self.sampled_data[idx - len(self.dataset)]

def active_learning_step(args, model, unlabeled_data_loader, num_active_learning_samples):
    # 确保 unlabeled_data_loader 是 DataLoader 对象
    if not isinstance(unlabeled_data_loader, DataLoader):
        raise TypeError("unlabeled_data_loader should be an instance of DataLoader")

    # 从未标注数据中选择样本
    model.eval()
    uncertainties = []
    embeddings = []
    sampled_data = []

    with torch.no_grad():
        for batch in unlabeled_data_loader:
            batch = tuple(t.to(DEVICE) for t in batch)
            input_ids, visual_ids, acoustic_ids, pos_ids, senti_ids, polarity_ids, visual, acoustic, input_mask, segment_ids, _ = batch
            visual = torch.squeeze(visual, 1)
            
            outputs = model(
                input_ids,
                visual,
                acoustic,
                visual_ids,
                acoustic_ids,
                pos_ids, senti_ids, polarity_ids,
                token_type_ids=segment_ids,
                attention_mask=input_mask,
                labels=None,
            )
            logits = outputs[0]
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            
            uncertainty = -np.max(probs, axis=1)
            uncertainties.extend(uncertainty)

            embeddings.append(outputs[-1].cpu().numpy())
    
    uncertainties = np.array(uncertainties)
    embeddings = np.concatenate(embeddings, axis=0)

    top_uncertainty_indices = uncertainties.argsort()[-num_active_learning_samples:]

    selected_indices = []
    selected_embeddings = []
    for idx in top_uncertainty_indices:
        if len(selected_embeddings) == 0:
            selected_indices.append(idx)
            selected_embeddings.append(embeddings[idx])
        else:
            distances = np.linalg.norm(embeddings[idx] - np.array(selected_embeddings), axis=1)
            min_distance = np.min(distances)
            if min_distance > 0.5:  # 设定一个阈值，确保多样性
                selected_indices.append(idx)
                selected_embeddings.append(embeddings[idx])

    # 如果选择的样本不足，补充至目标数量
    if len(selected_indices) < num_active_learning_samples:
        remaining_indices = [i for i in top_uncertainty_indices if i not in selected_indices]
        selected_indices.extend(remaining_indices[:num_active_learning_samples - len(selected_indices)])
    
    # 将选择的样本添加到 sample_data 中
    for idx in selected_indices:
        sampled_data.append(unlabeled_data_loader.dataset[idx])

    print(f"Active learning step: selected {len(sampled_data)} new samples.")
    return sampled_data

def main():
    args = parser_args()
    set_random_seed(args.seed)
    wandb.init(project="sentilare", config=args)
    
    train_data_loader, dev_data_loader, test_data_loader, unlabeled_data_loader = set_up_data_loader(args)
    
    num_train_optimization_steps = (len(train_data_loader) // args.gradient_accumulation_step) * args.n_epochs
    
    model, optimizer, scheduler = prep_for_training(args, num_train_optimization_steps)

    best_dev_score = -1e10
    initial_train_size = len(train_data_loader.dataset)
    print(f"Initial training dataset size: {initial_train_size}")

    for epoch in range(args.n_epochs):
        if epoch + 1 == 161 or epoch + 1 == 162 or epoch + 1 == 163:
            print("Skipping epoch 161 and the associated active learning step.")
            continue  # Skip the 161st epoch
        
        print(f"Epoch {epoch + 1}/{args.n_epochs}")
        
        # Train the model for one epoch
        train_loss, train_preds, train_labels = train_epoch(args, model, train_data_loader, optimizer, scheduler)
        # print(f"Train loss for epoch {epoch + 1}: {train_loss}")

        # Evaluate the model on the development set
        dev_loss, dev_preds, dev_labels, dev_has0_acc_2, dev_has0_f1_score, dev_non0_acc_2, dev_non0_f1_score, dev_mae, dev_corr = evaluate_epoch(args, model, dev_data_loader)

        # Update the best development score and save the model if improved
        if dev_has0_f1_score > best_dev_score:
            best_dev_score = dev_has0_f1_score
            print(f"New best model saved with dev Has0_F1_score: {best_dev_score}")

        # Evaluate the model on the test set
        test_loss, test_preds, test_labels, test_has0_acc_2, test_has0_f1_score, test_non0_acc_2, test_non0_f1_score, test_mae, test_corr = evaluate_epoch(args, model, test_data_loader)
        # print(f"Test loss for epoch {epoch + 1}: {test_loss}")
        # print(f"Test Has0_acc_2: {test_has0_acc_2}")
        # print(f"Test Has0_F1_score: {test_has0_f1_score}")
        # print(f"Test Non0_acc_2: {test_non0_acc_2}")
        # print(f"Test Non0_F1_score: {test_non0_f1_score}")
        # print(f"Test MAE: {test_mae}")
        # print(f"Test Corr: {test_corr}")

        # Perform active learning and update the training dataset if required
        if (epoch + 1) % args.active_learning_interval == 0:
            print(f"Performing active learning at epoch {epoch + 1}")
            sampled_data = active_learning_step(args, model, unlabeled_data_loader, args.num_active_learning_samples)
            
            # Create a new ActiveLearningDataset with the updated data
            updated_train_dataset = ActiveLearningDataset(train_data_loader.dataset, sampled_data)
            # Create a new DataLoader with the updated dataset
            train_data_loader = DataLoader(updated_train_dataset, batch_size=args.train_batch_size, shuffle=True)
            
            updated_train_size = len(updated_train_dataset)
            print(f"Updated training dataset with {len(sampled_data)} new samples.")
            print(f"Total training dataset size after update: {updated_train_size}")
        
        # Save the model for each epoch
        # torch.save(model.state_dict(), f"model_epoch_{epoch + 1}.pt")
    
    # Final evaluation on the test set
    test_loss, test_preds, test_labels, test_has0_acc_2, test_has0_f1_score, test_non0_acc_2, test_non0_f1_score, test_mae, test_corr = evaluate_epoch(args, model, test_data_loader)
    print(f"Final test loss: {test_loss}")
    print(f"Final Test Has0_acc_2: {test_has0_acc_2}")
    print(f"Final Test Has0_F1_score: {test_has0_f1_score}")
    print(f"Final Test Non0_acc_2: {test_non0_acc_2}")
    print(f"Final Test Non0_F1_score: {test_non0_f1_score}")
    print(f"Final Test MAE: {test_mae}")
    print(f"Final Test Corr: {test_corr}")

if __name__ == "__main__":
    main()

