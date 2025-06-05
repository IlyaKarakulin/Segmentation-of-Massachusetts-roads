import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms.functional as TF

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score

from dataset import SegmentDataset
from seg_hrnet import HighResolutionNet


class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        num = targets.size(0)
        probs = torch.sigmoid(logits)

        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)

        intersection = (m1 * m2)

        score = 2 * (intersection.sum(1) + self.smooth) / (m1.sum(1) + m2.sum(1) + self.smooth)
        score = 1 - score.sum() / num

        return score
    
# class SoftIoULoss(nn.Module):
#     def __init__(self, smooth=1):
#         super().__init__()
#         self.smooth = smooth

#     def forward(self, logits, targets):
#         num = targets.size(0)
#         probs = torch.sigmoid(logits)
#         m1 = probs.view(num, -1)
#         m2 = targets.view(num, -1)
#         intersection = (m1 * m2)

#         score = (intersection.sum(1) + self.smooth) / (m1.sum(1) + m2.sum(1) - intersection.sum(1) + self.smooth)
#         score = 1 - score.sum() / num
#         return score
    

# class BoundaryLoss (nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, logits, sdm):
#         probs = torch.sigmoid(logits)
#         clipped_sdm = torch.relu(sdm)

#         norm = clipped_sdm.mean(dim=[2,3], keepdim=True) + 1e-6
#         weighted = probs * clipped_sdm / norm

#         score = weighted.mean()
#         return score


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.8):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # Вес для положительного класса (дороги)

    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        
        # Балансировка классов
        alpha_weight = torch.where(
            targets == 1, 
            self.alpha, 
            1 - self.alpha
        )
        
        focal_loss = alpha_weight * focal_weight * bce_loss
        return focal_loss.mean()



class Segmentator():
    def __init__(self, device='cpu', conf=None):
        self.device = device
       
        self.model = HighResolutionNet(conf).to(self.device)

        def count_parameters(model):
            return sum(p.numel() for p in model.parameters())

        total_params = count_parameters(self.model)
        print(f"Всего параметров: {total_params:,}")

        self.writer = None 
        self.num_workers = 6
        self.pin_memory = True

    def train(self, path_to_train: str, path_to_val: str, num_epoch=100, batch_size=64, acc_step=1, lr=0.001):

        segment_dataset = SegmentDataset(path_to_train, 'train')
        dataloader_train = data.DataLoader(segment_dataset, batch_size=batch_size, shuffle=True, 
                                           num_workers=self.num_workers, pin_memory=self.pin_memory)
        
        segment_val = SegmentDataset(path_to_val, 'val')
        dataloader_val = data.DataLoader(segment_val, batch_size=1, shuffle=False,
                                          num_workers=self.num_workers, pin_memory=self.pin_memory)
        

        self.optimizer = optim.AdamW(params=self.model.parameters(), lr=lr, weight_decay=0)

        gamma = 2
        alpha = 0.8
        loss_1 = FocalLoss(gamma, alpha)

        # loss_1 = nn.BCEWithLogitsLoss()
        loss_2 = SoftDiceLoss()

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=6)

        log_dir = f'meta_data/HRNet/bs={batch_size}*{acc_step}_res={segment_dataset.size}_(1,2,0,0)_Focal_gamma={gamma}, alpha={alpha}'
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     self.optimizer,
        #     T_max=num_epoch,
        #     eta_min=1e-6
        # )

        self.model.train()
        best_val_iou = 0.0


        for count_epoch in range(num_epoch):
            train_metrics = self.run_epoch(count_epoch, dataloader_train, self.optimizer, loss_1, loss_2, acc_step, is_train=True)
            val_metrics = self.run_epoch(count_epoch, dataloader_val, None, loss_1, loss_2, acc_step, is_train=False)
            # train_metrics = self.run_epoch(count_epoch, dataloader_train, self.optimizer, loss_1, None, acc_step, is_train=True)
            # val_metrics = self.run_epoch(count_epoch, dataloader_val, None, loss_1, None, acc_step, is_train=False)

            self.scheduler.step(val_metrics['Val_Loss'])
            current_lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('Learning Rate', current_lr, count_epoch)

            self.__log_metrics(train_metrics, val_metrics, count_epoch)
            # self.__log_weights_and_grads(count_epoch)
                        
            if val_metrics["Val_IoU"] > best_val_iou:
                self.__save_model("best")
                best_val_iou = val_metrics["Val_IoU"]
                
            print(
                f"Lr = {round(current_lr, 4)}\n"
                f"Train  Loss: {train_metrics['Train_Loss']:.4f} | "
                f"IoU: {train_metrics['Train_IoU']:.4f}\n"
                f"Val    Loss: {val_metrics['Val_Loss']:.4f} | "
                f"IoU: {val_metrics['Val_IoU']:.4f} | "
                f"P: {val_metrics['Val_P']:.4f} | "
                f"R: {val_metrics['Val_R']:.4f}"
            )

            self.__save_model("last")

        self.writer.close()

    def run_epoch(self, epoch_num, dataloader, optimizer, loss1, loss2, acc_step, is_train=True):
        if is_train:
            self.model.train()
            mode_str = "Train"
        else:
            self.model.eval()
            mode_str = "Val"

        epoch_loss = 0.0
        all_iou_scores = []
        total_samples = 0 
        
        accumulated_steps = 0
        
        progress_bar = tqdm(dataloader, desc=f"{epoch_num+1}", leave=is_train, ncols=100)

        for i, (x_batch, y_batch) in enumerate(progress_bar):
            x_batch = x_batch.to(self.device, non_blocking=self.pin_memory)
            y_batch = y_batch.to(self.device, non_blocking=self.pin_memory)
            batch_size = x_batch.size(0)
            total_samples += batch_size

            if is_train:
               
                with torch.set_grad_enabled(True):
                    predictions = self.model(x_batch)
                    loss1_value = loss1(predictions, y_batch)

                    if loss2:
                        loss2_value = loss2(predictions, y_batch)
                    else:
                        loss2_value = 0

                    loss_unscaled = loss1_value + loss2_value 
                    
                    loss = loss_unscaled / acc_step
                    loss.backward()
                    accumulated_steps += 1

                if accumulated_steps % acc_step == 0 or i == len(dataloader) - 1:
                    # nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                   
                    accumulated_steps = 0
            else:
               
                with torch.no_grad():
                    predictions = self.model(x_batch)
                    loss1_value = loss1(predictions, y_batch)

                    if loss2:
                        loss2_value = loss2(predictions, y_batch)
                    else:
                        loss2_value = 0
                    loss_unscaled = loss1_value + loss2_value

            precision_arr = []
            recall_arr = []
           
            with torch.no_grad():
                probs = torch.sigmoid(predictions)
                preds_binary = (probs > 0.5).float()
                
                intersection = (preds_binary * y_batch).sum(dim=[1,2,3])
                union = preds_binary.sum(dim=[1,2,3]) + y_batch.sum(dim=[1,2,3]) - intersection
                
                smooth_iou = 1e-6
                iou_batch = (intersection + smooth_iou) / (union + smooth_iou)
                all_iou_scores.extend(iou_batch.cpu().tolist())
                
                if not is_train:
                    y_true = y_batch.cpu().numpy().flatten()
                    y_pred = preds_binary.cpu().numpy().flatten()
                    
                    y_pred = (y_pred > 0.5).astype(np.uint8)

                    precision_arr.append(precision_score(y_true, y_pred, zero_division=True))
                    recall_arr.append(recall_score(y_true, y_pred, zero_division=True))

           
            epoch_loss += loss_unscaled.item() * batch_size
            
            if is_train:
                progress_bar.set_postfix(loss=f"{loss_unscaled.item():.4f}")
        
       
        avg_epoch_loss = epoch_loss / total_samples
        metrics = {f"{mode_str}_Loss": avg_epoch_loss}

        avg_iou = np.mean(all_iou_scores) if all_iou_scores else 0.0
        metrics[f"{mode_str}_IoU"] = avg_iou

        if not is_train:
            metrics[f"{mode_str}_P"] = sum(precision_arr) / len(precision_arr)
            metrics[f"{mode_str}_R"] = sum(recall_arr) / len(recall_arr)
        
        return metrics    
    
    def __log_metrics(self, train_metrics, val_metrics, epoch):
        for metric, value in train_metrics.items():
            self.writer.add_scalar(f'Train/{metric.split("_")[1]}', value, epoch)

        for metric, value in val_metrics.items():
            self.writer.add_scalar(f'Val/{metric.split("_")[1]}', value, epoch)

    def __save_model(self, name):
        model_dir = './meta_data/models'
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"{name}.pth")
        
        state = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()
        }

        torch.save(state, model_path)

    def load_model(self, path_to_checkpoint):
        if not os.path.exists(path_to_checkpoint):
            print(f"Checkpoint file not found: {path_to_checkpoint}")
            return

        checkpoint = torch.load(path_to_checkpoint, map_location=torch.device(self.device), weights_only=True)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.model.eval()
        print(f"Model loaded from {path_to_checkpoint}")

    def test(self, path_to_test_data: str, batch_size=1, save_dir="test_results"):
        os.makedirs(save_dir, exist_ok=True)
        test_dataset = SegmentDataset(path_to_test_data, 'test')
        print(f"Test dataset size: {test_dataset.__len__()} samples")

        test_loader = data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=self.pin_memory
        )

        self.model.eval()
        all_ious = []

        with torch.no_grad():
            for batch_idx, (x_batch, y_batch) in enumerate(tqdm(test_loader, desc="Testing", ncols=100)):
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                predictions, x1, x2, x3, x4 = self.model(x_batch)
                # print()
                # print(x1.size())
                # print(x2.size())
                # print(x3.size())
                # print(x4.size())

                probs = torch.sigmoid(predictions)
                preds_binary = (probs > 0.05)

                intersection = (preds_binary * y_batch).sum(dim=[1, 2, 3])
                union = preds_binary.sum(dim=[1, 2, 3]) + y_batch.sum(dim=[1, 2, 3]) - intersection
                iou = (intersection + 1e-6) / (union + 1e-6)
                all_ious.extend(iou.cpu().tolist())
        
                if batch_idx < 10:
                    self.visualize_all_outputs(
                        x_batch, 
                        y_batch, 
                        preds_binary,
                        [x1, x2, x3, x4],
                        save_path=os.path.join(save_dir, f"batch_{batch_idx}.png")
                    )

        mean_iou = np.mean(all_ious)
        print(f"Mean IoU on test: {mean_iou:.4f}")

        return []

    def visualize_all_outputs(self, x_batch, y_batch, final_pred, intermediates, save_path=None, num_examples=4):
        batch_size = min(x_batch.size(0), num_examples)
        num_layers = len(intermediates) + 3
        
        fig, axs = plt.subplots(batch_size, num_layers, figsize=(4*num_layers, 4*batch_size))
        
        if batch_size == 1:
            axs = [axs]
        
        layer_names = ["Input", "Ground Truth", "Final Prediction"] + [f"Layer {i+1}" for i in range(len(intermediates))]
        
        for i in range(batch_size):
            # Оригинальное изображение
            img = x_batch[i].detach().cpu()
            img = TF.to_pil_image(img)
            axs[i][0].imshow(img)
            axs[i][0].set_title(layer_names[0])
            axs[i][0].axis('off')
            
            # Ground Truth
            gt = y_batch[i, 0].detach().cpu().numpy()
            axs[i][1].imshow(gt, cmap='gray')
            axs[i][1].set_title(layer_names[1])
            axs[i][1].axis('off')
            
            # Финальный prediction
            pr = final_pred[i, 0].detach().cpu().numpy()
            axs[i][2].imshow(pr, cmap='gray')
            axs[i][2].set_title(layer_names[2])
            axs[i][2].axis('off')


            x1 = intermediates[0][i, 0].detach().cpu().numpy()
            axs[i][3].imshow(x1, cmap='gray')
            axs[i][3].set_title(layer_names[3])
            axs[i][3].axis('off')

            x2 = intermediates[1][i, 0].detach().cpu().numpy()
            axs[i][4].imshow(x2, cmap='gray')
            axs[i][4].set_title(layer_names[4])
            axs[i][4].axis('off')

            x3 = intermediates[2][i, 0].detach().cpu().numpy()
            axs[i][5].imshow(x3, cmap='gray')
            axs[i][5].set_title(layer_names[5])
            axs[i][5].axis('off')

            x4 = intermediates[3][i, 0].detach().cpu().numpy()
            axs[i][6].imshow(x4, cmap='gray')
            axs[i][6].set_title(layer_names[6])
            axs[i][6].axis('off')
            
            
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()

    # def visualize_batch(self, x_batch, y_batch, preds_binary, save_path=None, num_examples=4):
    #     batch_size = min(x_batch.size(0), num_examples)
    #     fig, axs = plt.subplots(batch_size, 3, figsize=(15, 3 * batch_size))
        
    #     if batch_size == 1:
    #         axs = [axs]
        
    #     for i in range(batch_size):
    #         img = x_batch[i].detach().cpu()
    #         img = TF.to_pil_image(img)
    #         gt = y_batch[i, 0].detach().cpu().numpy()
    #         pr = preds_binary[i, 0].detach().cpu().numpy()

    #         axs[i][0].imshow(img)
    #         axs[i][0].set_title('Input')
    #         axs[i][0].axis('off')

    #         axs[i][1].imshow(gt, cmap='gray')
    #         axs[i][1].set_title('Ground Truth')
    #         axs[i][1].axis('off')

    #         axs[i][2].imshow(pr, cmap='gray')
    #         axs[i][2].set_title('Prediction')
    #         axs[i][2].axis('off')

    #     plt.tight_layout()
    #     if save_path:
    #         plt.savefig(save_path)
    #         plt.close(fig)
    #     else:
    #         plt.show()