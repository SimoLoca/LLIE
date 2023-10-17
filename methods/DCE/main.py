import argparse
import torch
from torch.utils import data as data
import os
from glob import glob
import torchvision
from DCE_Net import DCE_Net
from dataset import Dataset
from train_utils import train_eval
import numpy as np
import random
import matplotlib.pyplot as plt


def parse():
    parser = argparse.ArgumentParser(
      description='Train or Eval Zero-DCE for low light enhancement')
    parser.add_argument('--train', choices=['True', 'False'],
                        help='Train flag: if false we eval the model.')
    parser.add_argument('--ckpt', default='', type=str,
                        help='path to a checkpoint')
    parser.add_argument('--img', default='', type=str,
                        help='path to an image to enhance')
    return parser.parse_args()


def set_seeds(seed, dev):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if dev == 'cuda':
        # training: disable cudnn benchmark to ensure the reproducibility
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # this is needed for CUDA >= 10.2
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True


def training(dataset_path, dev):
    image_files = glob(f'{dataset_path}/our485/low/*.png')
    print("files: ", len(image_files))

    random.shuffle(image_files)
    split_ratio = 0.8
    split_index = int(len(image_files) * split_ratio)

    # Split the image files into train and validation sets
    train_image_files = image_files[:split_index]
    val_image_files = image_files[split_index:]
    print("train files: ", len(train_image_files))
    print("val files: ", len(val_image_files))

    train_dataset = Dataset(image_files=train_image_files, image_size=256)
    train_dataloader = data.DataLoader(
            train_dataset, batch_size=8, shuffle=True,
            num_workers=8, pin_memory=True, drop_last=True
    )
    print("dataloader len: ", len(train_dataloader))

    val_dataset = Dataset(image_files=val_image_files, image_size=256)
    val_dataloader = data.DataLoader(
            val_dataset, batch_size=8, shuffle=False,
            num_workers=8, pin_memory=True
    )
    print("dataloader len: ", len(val_dataloader))

    model = DCE_Net()
    model.to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, mode='min', patience=10
    )
    
    train_eval(model, 100, train_dataloader, val_dataloader, optimizer, dev, scheduler)

    

@torch.inference_mode()
def eval(ckpt, img, dev):
    model = DCE_Net()
    model.load_state_dict(torch.load(ckpt)['model'])
    model.to(dev)
    print("Model loaded...")

    image = torchvision.io.read_image(img)/255.
    image = torchvision.transforms.functional.resize(
        image,
        (256, 256)
        )
    x = image.unsqueeze(0).to(dev)
    print(x.shape)
    
    enhanced, _ = model(x)
    enhanced = enhanced.squeeze().permute(1, 2, 0).cpu().numpy()

    # Plot the original image
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(image.permute(1, 2, 0).numpy())
    
    # Plot the enhanced image
    plt.subplot(1, 2, 2)
    plt.title('Enhanced Image')
    plt.imshow(enhanced)

    # Show the plot
    plt.show()



if __name__ == "__main__":
    args = parse()

    device = "cuda" if torch.cuda.is_available() else "cpu"
 
    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    set_seeds(42, device)
    
    print("Device: ", device)
    if args.train == "True":
        training("/home/user/Desktop/LOLdataset/", device)
    else:
        eval(args.ckpt, args.img, device)

