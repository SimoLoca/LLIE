import torch
import numpy as np
from tqdm import tqdm
import os, shutil
from losses import (
    color_constancy_loss,
    exposure_control_loss,
    spatial_consistency_loss,
    illumination_smoothness_loss
)


def save_ckpt(state: dict, is_best: bool, epoch: int, ckpt_dir: str):
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    filename = os.path.join(ckpt_dir, f'epoch_{str(epoch)}_ckpt.pth')
    torch.save(state, filename)
    if is_best:
        print(f'[BEST MODEL] Saving best model, obtained on epoch = {epoch + 1}')
        shutil.copy(filename, os.path.join(ckpt_dir, 'best_model.pth'))


def train_eval(
        model: torch.nn.Module, 
        epochs: int, 
        train_dataloader, 
        val_dataloader, 
        optimizer, 
        dev: str, 
        scheduler=None
    ):
    best_loss = float('inf')

    for epoch in range(1, epochs + 1):
        print('Epoch {}/{}'.format(epoch, epochs))
        model.train()
        loss_history = []
        for i, lowlight_imgs in tqdm(enumerate(train_dataloader)):
            low_light_images = lowlight_imgs.to(dev)

            optimizer.zero_grad()

            enhanced, stack = model(low_light_images)

            loss_tv = 200 * illumination_smoothness_loss(stack)
            loss_spa = torch.mean(
                spatial_consistency_loss(enhanced, low_light_images)
            )
            loss_col = 5 * torch.mean(color_constancy_loss(enhanced))
            loss_exp = 10 * torch.mean(exposure_control_loss(enhanced))
            loss = loss_tv + loss_spa + loss_col + loss_exp
            loss_history.append(loss.item())

            loss.backward()
            optimizer.step()
            if scheduler is not None: scheduler.step(loss)
        print(f"\tTrain Loss at epoch {epoch}: {np.mean(loss_history)}")

        model.eval()
        val_loss_history = []
        with torch.inference_mode():
            for i, lowlight_imgs in tqdm(enumerate(val_dataloader)):
                low_light_images = lowlight_imgs.to(dev)
                enhanced, stack = model(low_light_images)

                loss_tv = 200 * illumination_smoothness_loss(stack)
                loss_spa = torch.mean(
                    spatial_consistency_loss(enhanced, low_light_images)
                )
                loss_col = 5 * torch.mean(color_constancy_loss(enhanced))
                loss_exp = 10 * torch.mean(exposure_control_loss(enhanced))
                loss = loss_tv + loss_spa + loss_col + loss_exp
                val_loss_history.append(loss.item())
        
        val_loss = np.mean(val_loss_history)
        print(f"\tVal Loss at epoch {epoch}: {val_loss}")

        is_best = val_loss < best_loss
        best_loss = min(val_loss, best_loss)

        if epoch % 5 == 0 or is_best:
            save_ckpt(
                {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_loss': val_loss,
                'loss_history': loss_history,
                }, 
                is_best, epoch, './ckpt'
            )
