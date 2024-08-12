import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
import multiprocessing
import random
import numpy as np
from tqdm.auto import tqdm
from torchinfo import summary
from torchvision.datasets import STL10

from simple_ijepa.ijepa import IJEPA
from simple_ijepa.model import VisionTransformer

# from relic.utils import accuracy, get_dataset, get_encoder
from simple_ijepa.stl10_eval import STL10Eval
from simple_ijepa.utils import training_transforms
from simple_ijepa.dataset import MaskedImageDataset, collate_fn

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


# cosine EMA schedule (increase from tau_base to one) as defined in https://arxiv.org/abs/2010.07922
# k -> current training step, K -> maximum number of training steps
def update_gamma(k, K, tau_base):
    k = torch.tensor(k, dtype=torch.float32)
    K = torch.tensor(K, dtype=torch.float32)

    tau = 1 - (1 - tau_base) * (torch.cos(torch.pi * k / K) + 1) / 2
    return tau.item()


def train_ada_mim(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dim = 768
    image_size = 96
    patch_size = 8
    depth = 6
    heads = 6
    mlp_dim = dim * 2

    encoder = VisionTransformer(image_size=image_size, patch_size=patch_size, dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim)

    num_targets = 4
    ada_mim = IJEPA(encoder,
                    hidden_emb_dim=dim,
                    patch_size=patch_size,
                    num_targets=num_targets)

    if args.ckpt_path:
        model_state = torch.load(args.ckpt_path)
        ada_mim.load_state_dict(model_state)
    ada_mim = ada_mim.to(device)

    # summary(ada_mim, input_size=(2, 3, 96, 96))

    params = list(ada_mim.online_encoder.parameters()) + [ada_mim.mask_token] + list(ada_mim.prediction_head.parameters())
    optimizer = torch.optim.Adam(params,
                                 lr=args.learning_rate,
                                 weight_decay=args.weight_decay)

    stl10_ds = STL10("data/",
                split='unlabeled',
                download=True,
                transform=training_transforms((image_size, image_size)))
    num_patches = int((image_size // patch_size)) ** 2
    dataset = MaskedImageDataset(stl10_ds, num_patches=num_patches, num_targets=num_targets)
    train_loader = DataLoader(dataset,
                              batch_size=args.batch_size,
                              num_workers=multiprocessing.cpu_count() - 8,
                              drop_last=True,
                              pin_memory=True,
                              collate_fn=collate_fn,
                              shuffle=True)

    scaler = GradScaler(enabled=args.fp16_precision)

    stl10_eval = STL10Eval()
    total_num_steps = (len(train_loader) *
                       (args.num_epochs + 2)) - args.update_gamma_after_step
    gamma = args.gamma
    global_step = 0
    total_loss = 0.0
    for epoch in range(args.num_epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader,
                            desc=f"Epoch {epoch+1}/{args.num_epochs}")
        for step, (images, context_indices, target_indices_list) in enumerate(progress_bar):
            images = images.to(device)
            with autocast(enabled=args.fp16_precision): 
                loss = ada_mim(images, context_indices, target_indices_list)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if global_step > args.update_gamma_after_step and global_step % args.update_gamma_every_n_steps == 0:
                ada_mim.update_params(gamma)
                gamma = update_gamma(global_step, total_num_steps, args.gamma)

            if global_step <= args.update_gamma_after_step:
                ada_mim.copy_params()

            total_loss += loss.item()
            epoch_loss += loss.item()
            avg_loss = total_loss / (global_step + 1)
            ep_loss = epoch_loss / (step + 1)


            current_lr = optimizer.param_groups[0]['lr']
            progress_bar.set_description(
                f"Epoch {epoch+1}/{args.num_epochs} | "
                f"Step {global_step+1} | "
                f"Epoch Loss: {ep_loss:.7f} |"
                f"Total Loss: {avg_loss:.7f} |"
                f"EMA gamma: {gamma:.6f} |"
                f"Lr: {current_lr:.6f}")

            global_step += 1
            if global_step % args.log_every_n_steps == 0:
                torch.save(ada_mim.state_dict(),
                           f"{args.save_model_dir}/training_model.pth")
                ada_mim.save_encoder(f"{args.save_model_dir}/encoder.pth")

            if global_step % (args.log_every_n_steps * 5) == 0:
                stl10_eval.evaluate(ada_mim)
                print("!" * 100)
