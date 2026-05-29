import numpy as np
import torch
from torch import nn
from torch.nn import BatchNorm1d

class Quantizer(nn.Module):
    def __init__(self, embed_dim, num_embed, decay, threshold, eps=1e-5):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_embed = num_embed
        self.decay = decay
        self.threshold = threshold
        self.eps = eps
        embed = torch.randn(embed_dim, num_embed)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(num_embed))
        self.register_buffer("embed_mean", embed.clone())

    def forward(self, input):
        flatten = input.reshape(-1, self.embed_dim)
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        euclidean_dist = torch.sqrt(dist + self.eps)
        pseudo_probs = torch.softmax(-euclidean_dist, dim=1)
        sorted_indices = euclidean_dist.argsort(dim=1)

        struct_dtype = [
            ("rank", np.int32),
            ("embed_id", np.int32),
            ("eu_dist", np.float32),
            ("pseudo_probs", np.float32),
        ]
        embedding_info = np.empty((input.shape[0], input.shape[1]), dtype=object)
        for i in range(input.shape[0] * input.shape[1]):
            batch_idx, time_idx = divmod(i, input.shape[1])
            info_array = np.zeros(self.num_embed, dtype=struct_dtype)
            order = sorted_indices[i]
            info_array["rank"] = np.arange(self.num_embed, dtype=np.int32)
            info_array["embed_id"] = order.detach().cpu().numpy().astype(np.int32)
            info_array["eu_dist"] = euclidean_dist[i, order].detach().cpu().numpy().astype(np.float32)
            info_array["pseudo_probs"] = pseudo_probs[i, order].detach().cpu().numpy().astype(np.float32)
            embedding_info[batch_idx, time_idx] = info_array

        _, embed_ind = (-dist).max(1)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = torch.nn.functional.embedding(embed_ind, self.embed.transpose(0, 1))

        diff = (quantize.detach() - input).pow(2).mean()
        quantize = input + (quantize - input).detach()
        return quantize, diff, embed_ind, embedding_info

class Encoder(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers, conv_dims, kernel_sizes, strides, p, mask_flag):
        super().__init__()
        self.num_layers = num_layers
        self.mask_flag = mask_flag
        self.input_dim = input_dim
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout1d(p)

        if self.mask_flag in [1, 2]:
            self.intra_mask_1 = nn.Conv1d(input_dim, input_dim, kernel_size=3, stride=1, padding=1)
            self.bn_intra_mask_1 = nn.BatchNorm1d(input_dim)
            self.intra_mask_2 = nn.Conv1d(input_dim, input_dim, kernel_size=3, stride=1, padding=1)
            self.bn_intra_mask_2 = nn.BatchNorm1d(input_dim)
            self.pre_mask = nn.Sequential(
                self.intra_mask_1,
                self.bn_intra_mask_1,
                self.relu,
                self.intra_mask_2,
                self.bn_intra_mask_2,
                self.relu,
            )

        conv1_in = input_dim * 2 if self.mask_flag in [1, 2] else input_dim
        self.conv1 = nn.Conv1d(conv1_in, input_dim, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(input_dim)
        self.conv2 = nn.Conv1d(input_dim, input_dim * 2, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(input_dim * 2)
        self.conv3 = nn.Conv1d(input_dim * 2, input_dim * 4, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(input_dim * 4)

        if self.mask_flag in [1, 2]:
            self.conv4 = nn.Conv1d(input_dim * 4, input_dim * 4, kernel_size=3, stride=1, padding=1)
            self.bn4 = nn.BatchNorm1d(input_dim * 4)
            self.conv5 = nn.Conv1d(input_dim * 4, input_dim * 6, kernel_size=3, stride=1, padding=1)
            self.bn5 = nn.BatchNorm1d(input_dim * 6)
            self.conv6 = nn.Conv1d(input_dim * 6, input_dim * 8, kernel_size=3, stride=1, padding=1)
            self.bn6 = nn.BatchNorm1d(input_dim * 8)
            self.conv_layers = nn.Sequential(
                self.conv1,
                self.bn1,
                self.relu,
                self.conv2,
                self.bn2,
                self.relu,
                self.conv3,
                self.bn3,
                self.relu,
                self.conv4,
                self.bn4,
                self.relu,
                self.conv5,
                self.bn5,
                self.relu,
                self.conv6,
                self.bn6,
                self.relu,
            )
        else:
            self.conv4 = nn.Conv1d(input_dim * 4, input_dim * 8, kernel_size=3, stride=1, padding=1)
            self.bn4 = nn.BatchNorm1d(input_dim * 8)
            self.conv_layers = nn.Sequential(
                self.conv1,
                self.bn1,
                self.relu,
                self.conv2,
                self.bn2,
                self.relu,
                self.conv3,
                self.bn3,
                self.relu,
                self.conv4,
                self.bn4,
                self.relu,
            )

    def forward(self, input, mask=None):
        if self.mask_flag in [1, 2]:
            assert mask is not None
            for layer in self.pre_mask:
                mask = layer(mask)
            input = torch.cat([input, mask], dim=1)
        else:
            assert mask is None
        for layer in self.conv_layers:
            input = layer(input)
        return input

class Decoder(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers, conv_dims, kernel_sizes, strides, p, mask_flag):
        super().__init__()
        self.num_layers = num_layers
        self.mask_flag = mask_flag
        self.input_dim = input_dim
        self.relu = nn.ReLU()
        self.id = nn.Identity()
        self.dropout = nn.Dropout1d(p)

        self.deconv1 = nn.ConvTranspose1d(output_dim, input_dim * 6, kernel_size=3, stride=1, padding=1)
        self.bn1 = BatchNorm1d(input_dim * 6)
        self.deconv2 = nn.ConvTranspose1d(input_dim * 6, input_dim * 4, kernel_size=3, stride=1, padding=1)
        self.bn2 = BatchNorm1d(input_dim * 4)
        self.deconv3 = nn.ConvTranspose1d(input_dim * 4, input_dim * 4, kernel_size=3, stride=1, padding=1)
        self.bn3 = BatchNorm1d(input_dim * 4)
        self.deconv4 = nn.ConvTranspose1d(input_dim * 4, input_dim * 2, kernel_size=3, stride=1, padding=1)
        self.bn4 = BatchNorm1d(input_dim * 2)
        self.deconv5 = nn.ConvTranspose1d(input_dim * 2, input_dim, kernel_size=3, stride=1, padding=1)
        self.bn5 = BatchNorm1d(input_dim)

        final_activation = self.relu if self.mask_flag == 2 else self.id
        self.deconv_layers = nn.Sequential(
            self.deconv1,
            self.bn1,
            self.relu,
            self.deconv2,
            self.bn2,
            self.relu,
            self.deconv3,
            self.bn3,
            self.relu,
            self.deconv4,
            self.bn4,
            self.relu,
            self.deconv5,
            self.bn5,
            final_activation,
        )

        if self.mask_flag == 2:
            self.intra_mask_1 = nn.Conv1d(input_dim, input_dim, kernel_size=3, stride=1, padding=1)
            self.bn_intra_mask_1 = nn.BatchNorm1d(input_dim)
            self.intra_mask_2 = nn.Conv1d(input_dim, input_dim, kernel_size=3, stride=1, padding=1)
            self.bn_intra_mask_2 = nn.BatchNorm1d(input_dim)
            self.pre_mask = nn.Sequential(
                self.intra_mask_1,
                self.bn_intra_mask_1,
                self.relu,
                self.intra_mask_2,
                self.bn_intra_mask_2,
                self.relu,
            )
            self.fine1 = nn.Conv1d(input_dim * 2, input_dim * 2, kernel_size=3, stride=1, padding=1)
            self.bn_fine1 = BatchNorm1d(input_dim * 2)
            self.fine2 = nn.Conv1d(input_dim * 2, input_dim, kernel_size=3, stride=1, padding=1)
            self.bn_fine2 = BatchNorm1d(input_dim)
            self.fine3 = nn.Conv1d(input_dim, input_dim, kernel_size=3, stride=1, padding=1)
            self.bn_fine3 = BatchNorm1d(input_dim)
            self.fine4 = nn.Conv1d(input_dim, input_dim, kernel_size=3, stride=1, padding=1)
            self.bn_fine4 = BatchNorm1d(input_dim)
            self.fine_tune_layers = nn.Sequential(
                self.fine1,
                self.bn_fine1,
                self.relu,
                self.fine2,
                self.bn_fine2,
                self.relu,
                self.fine3,
                self.bn_fine3,
                self.relu,
                self.fine4,
                self.bn_fine4,
                self.id,
            )

    def forward(self, input, mask=None):
        if self.mask_flag == 2:
            assert mask is not None
        else:
            assert mask is None
        for layer in self.deconv_layers:
            input = layer(input)
        if self.mask_flag == 2:
            for layer in self.pre_mask:
                mask = layer(mask)
            input = torch.cat([input, mask], dim=1)
            for layer in self.fine_tune_layers:
                input = layer(input)
        return input

class VQVAE(nn.Module):
    def __init__(
        self,
        num_features,
        embed_dim,
        num_embed,
        num_layers,
        conv_dims,
        kernel_sizes,
        strides,
        p,
        decay,
        threshold,
        mask_flag,
    ):
        super().__init__()
        self.mask_flag = mask_flag
        self.encoder = Encoder(num_features, embed_dim, num_layers, conv_dims, kernel_sizes, strides, p, mask_flag)
        self.decoder = Decoder(num_features, embed_dim, num_layers, conv_dims, kernel_sizes, strides, p, mask_flag)
        self.quantizer = Quantizer(embed_dim, num_embed, decay, threshold)

    def forward(self, input, mask=None):
        encoded = self.encoder(input, mask if self.mask_flag in [1, 2] else None)
        quantized, diff, indices, embedding_info = self.quantizer(encoded.permute(0, 2, 1))
        decoded = self.decoder(quantized.permute(0, 2, 1), mask if self.mask_flag == 2 else None)
        return decoded, diff, indices, embedding_info
