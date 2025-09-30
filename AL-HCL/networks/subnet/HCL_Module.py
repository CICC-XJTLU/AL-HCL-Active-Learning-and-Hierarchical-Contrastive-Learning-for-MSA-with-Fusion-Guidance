import torch 
import torch.nn as nn
import torch.nn.functional as F
from config.global_configs import *

class HCL_Total(nn.Module):
    def __init__(self, beta_shift_a=0.5, beta_shift_v=0.5, dropout_prob=0.2, name="", device='cpu'):
        super(HCL_Total, self).__init__()
        self.device = device
        self.visual_embedding = nn.Embedding(label_size + 1, TEXT_DIM, padding_idx=label_size).to(device)
        self.acoustic_embedding = nn.Embedding(label_size + 1, TEXT_DIM, padding_idx=label_size).to(device)
        self.hcl_module = HCLModule(temperature=0.1).to(device)
        self.cl3 = CL3(TEXT_DIM, num_modalities=3).to(device)
        self.hcl_3 = HCLModule_3(temperature=0.5).to(device)

    def forward(self, text_embedding, visual, acoustic, visual_ids, acoustic_ids):
        visual_ids = visual_ids.to(self.device)
        acoustic_ids = acoustic_ids.to(self.device)
        visual_ = self.visual_embedding(visual_ids)
        acoustic_ = self.acoustic_embedding(acoustic_ids)
        
        fusion = torch.cat((text_embedding, visual_, acoustic_), dim=2)
        fusion = self.cat_connect(fusion)

        # HCL模块
        # 第一层对比学习损失
        hcl_loss_text_audio = self.hcl_module(text_embedding, acoustic_)
        hcl_loss_text_visual = self.hcl_module(text_embedding, visual_)
        hcl_loss_audio_visual = self.hcl_module(acoustic_, visual_)
        hcl_loss_1 = hcl_loss_text_audio + hcl_loss_text_visual + hcl_loss_audio_visual
        hcl_loss_1 = hcl_loss_text_audio + hcl_loss_text_visual + hcl_loss_audio_visual

        # 第二层对比学习损失
        hcl_loss_fusion_audio = self.hcl_module(fusion, acoustic_)
        hcl_loss_fusion_visual = self.hcl_module(fusion, visual_)
        hcl_loss_fusion_text = self.hcl_module(fusion, text_embedding)
        hcl_loss_2 = hcl_loss_fusion_audio + hcl_loss_fusion_visual + hcl_loss_fusion_text

        # 第三层对比学习损失
        hcl_loss3_ta = self.hcl_3(text_embedding, acoustic_)
        hcl_loss3_tv = self.hcl_3(text_embedding, visual_)
        hcl_loss3_va = self.hcl_3(visual_, acoustic_)
        hcl_loss_3 = hcl_loss3_ta + hcl_loss3_tv + hcl_loss3_va
        # 总 HCL 损失
        total_hcl_loss = 0.2*hcl_loss_1 + 0.3*hcl_loss_2 + 0.5*hcl_loss_3

        return torch.tensor(total_hcl_loss)  # 直接返回总损失张量

    def get_loss(self, text_embedding, visual, acoustic, visual_ids, acoustic_ids):
        # 计算并返回总损失
        return self.forward(text_embedding, visual, acoustic, visual_ids, acoustic_ids)


class MBFModule(nn.Module):
    def __init__(self, input_size, output_size):
        super(MBFModule, self).__init__()
        self.conv1d = nn.Conv1d(input_size, output_size, kernel_size=1)
        self.conv2d = nn.Conv2d(output_size, output_size, kernel_size=(1, 1))
    
    def forward(self, features):
        outer_products = []
        for i in range(len(features)):
            for j in range(len(features)):
                if i != j:
                    outer_product = torch.einsum('ij,ik->jk', features[i], features[j])
                    outer_products.append(outer_product)

        # Concatenate outer products
        multimodal_matrix = torch.cat(outer_products, dim=0)
        
        # Ensure that the input to conv1d has the correct shape
        batch_size = multimodal_matrix.size(0)
        length = multimodal_matrix.size(1)
        
        refined_features = self.conv1d(multimodal_matrix.view(batch_size, 768, length // 768))  # Adjust accordingly
        return refined_features

class CL3(nn.Module):
    def __init__(self, input_size, num_modalities):
        super(CL3, self).__init__()
        self.mbf = MBFModule(input_size, input_size)
        self.temperature = 2
        self.num_modalities = num_modalities
    
    def forward(self, features):
        # features: a list of tensors for each modality
        I_m = [self.mbf(f).unsqueeze(0) for f in features]  # Apply MBF to each modality
        I_m = torch.cat(I_m, dim=0)  # Combine modalities
        
        # Compute the loss for each pair of modalities
        losses = []
        for m in range(self.num_modalities):
            for n in range(self.num_modalities):
                if m != n:
                    K_mn = torch.einsum('ijk,ijl->ikl', I_m[m], I_m[n])
                    
                    # Normalize K_mn to prevent numerical overflow
                    K_mn = K_mn / (K_mn.norm(dim=-1, keepdim=True) + 1e-8)  # Normalize each row
                    
                    K = K_mn @ K_mn.transpose(1, 2) / self.temperature  # Apply temperature
                    
                    # Add small constant to ensure numerical stability
                    K_diag = K.diagonal(dim1=1, dim2=2) + 1e-8
                    K_sum = torch.logsumexp(K, dim=2) + 1e-8  # Use logsumexp for stability
                    
                    # Calculate the loss
                    loss = -torch.mean(K_diag - K_sum)
                    
                    # Check for inf or NaN
                    if torch.isinf(loss).any() or torch.isnan(loss).any():
                        print(f"Inf or NaN detected in loss computation for modalities {m} and {n}")
                        print(f"K_mn: {K_mn}")
                        print(f"K: {K}")
                        print(f"K diagonal: {K_diag}")
                        print(f"Sum(exp(K)): {K_sum}")

                    losses.append(loss)
        
        L_final = sum(losses)
        return L_final
    
class HCLModule_3(nn.Module):
    def __init__(self, temperature):
        super(HCLModule_3, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()
        self.mbf = MBFModule(input_size=768, output_size=768)

    def forward(self, z1, z2):
        z1 = self.mbf(z1) # Apply MBF to each modality
        z2 = self.mbf(z2)  # Combine modalities
        # Normalize the embeddings
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)

        # 在最后一个维度上进行平均，得到 [64, 768] 的形状
        z1_avg = z1.mean(dim=1)  # [64, 768]
        z2_avg = z2.mean(dim=1)  # [64, 768]

        # Compute the similarity matrix
        similarity_matrix = torch.matmul(z1_avg, z2_avg.T) / self.temperature
        
        # Generate labels for contrastive learning
        labels = torch.arange(z1.size(0)).to(z1.device)
        
        # Calculate contrastive loss (InfoNCE)
        loss = self.criterion(similarity_matrix, labels)
        
        return loss

class HCLModule(nn.Module):
    def __init__(self, temperature):
        super(HCLModule, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, z1, z2):
        # Normalize the embeddings
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)

        # 在最后一个维度上进行平均，得到 [64, 768] 的形状
        z1_avg = z1.mean(dim=1)  # [64, 768]
        z2_avg = z2.mean(dim=1)  # [64, 768]

        # Compute the similarity matrix
        similarity_matrix = torch.matmul(z1_avg, z2_avg.T) / self.temperature
        
        # Generate labels for contrastive learning
        labels = torch.arange(z1.size(0)).to(z1.device)
        
        # Calculate contrastive loss (InfoNCE)
        loss = self.criterion(similarity_matrix, labels)
        
        return loss
