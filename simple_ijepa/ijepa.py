import torch
import copy
import torch.nn as nn
import torch.nn.functional as F

from simple_ijepa.utils import trunc_normal_
from simple_ijepa.transformer import  VisionTransformer


class IJEPA(torch.nn.Module):

    def __init__(self,
                 encoder,
                 hidden_emb_dim=512,
                 img_size=96,
                 patch_size=8,
                 num_targets=4,
                 predictor_depth=6,
                 predictor_heads=6):
        super(IJEPA, self).__init__()

        self.patch_size = patch_size
        self.img_size = img_size
        self.num_targets = num_targets

        self.context_encoder = encoder

        self.target_encoder = copy.deepcopy(self.context_encoder)
        self.target_encoder.requires_grad_(False)

        self.predictor = VisionTransformer(image_size=96,
                                           patch_size=patch_size,
                                           dim=hidden_emb_dim,
                                           depth=predictor_depth,
                                           heads=predictor_heads,
                                           mlp_dim=hidden_emb_dim * 2)

        self.mask_token = nn.Parameter(torch.zeros(1, hidden_emb_dim))
        trunc_normal_(self.mask_token, mean=0., std=.02)


    def forward(self, img, context_indices_list, target_indices_list):
        # Target model processes all patches
        with torch.no_grad():
            target_features = self.target_encoder(img).detach()
            target_features = F.layer_norm(target_features, (target_features.size(-1),))

        # Extract patches from the image
        patches = self.context_encoder.to_patch_embedding(img)
        B, num_patches, _ = patches.size()

        # Extract context features
        context_indices_list = torch.tensor(context_indices_list).to(patches.device)
        context_features = self.context_encoder(patches, patchify=False, mask=context_indices_list)

        # Apply positional embeddings to context features
        pos_emb = self.context_encoder.pos_embedding.to(context_features.device)
        context_pos_embed = torch.stack([pos_emb[context_indices_list[i], :] for i in range(B)])
        context_features += context_pos_embed

        min_target_size = min(min(len(indices) for indices in target_list) for target_list in target_indices_list)
        losses = []
        for j in range(self.num_targets):
            # Init mask tokens and apply pos encodings
            target_indices_batch = torch.stack([torch.tensor(target_indices_list[i][j], device=img.device) for i in range(B)])
            mask_tokens = self.mask_token.expand(B, min_target_size, -1).clone()
            mask_pos_embed = torch.stack([pos_emb[target_indices_batch[i], :] for i in range(B)])
            mask_tokens += mask_pos_embed

            # Concat context_features and mask_tokens, obtain predictions for masked target blocks
            predictor_input = torch.cat([context_features, mask_tokens], dim=1)
            predictor_output = self.predictor(predictor_input, patchify=False, pos_embed=False)

            predictor_output = F.layer_norm(predictor_output, (predictor_output.size(-1),))
            preds = predictor_output[:, -min_target_size:]
            loss = F.mse_loss(preds, torch.stack([target_features[i, target_indices_list[i][j], :] for i in range(B)]))
            losses.append(loss)

        total_loss = torch.mean(torch.stack(losses))
        return total_loss

    def update_params(self, gamma):
        with torch.no_grad():
            valid_types = [torch.float, torch.float16]
            for o_param, t_param in self._get_params():
                if o_param.dtype in valid_types and t_param.dtype in valid_types:
                    t_param.data.lerp_(o_param.data, 1. - gamma)

            for o_buffer, t_buffer in self._get_buffers():
                if o_buffer.dtype in valid_types and t_buffer.dtype in valid_types:
                    t_buffer.data.lerp_(o_buffer.data, 1. - gamma)

    def copy_params(self):
        for o_param, t_param in self._get_params():
            t_param.data.copy_(o_param)

        for o_buffer, t_buffer in self._get_buffers():
            t_buffer.data.copy_(o_buffer)

    def save_encoder(self, path):
        torch.save(self.target_encoder.state_dict(), path)

    def _get_params(self):
        return zip(self.context_encoder.parameters(),
                   self.target_encoder.parameters())

    def _get_buffers(self):
        return zip(self.context_encoder.buffers(),
                   self.target_encoder.buffers())
