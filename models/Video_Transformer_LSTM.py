import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from typing import Tuple, Union, Optional, Callable

from timm.models.vision_transformer import Block
from einops import rearrange

def make_2tuple(x):
    """构建tuple元胞组"""
    if isinstance(x, tuple):
        assert len(x) == 2; return x
    assert isinstance(x, int)
    return (x, x)

# 数据预处理
image_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(256),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class PatchEmbed(nn.Module):
    """定义输入特征点Patch层,对特征点进行分割处理
    num_points: 特征点数量
    embed_dim: 输出embedding维度
    """
    def __init__(
        self,
        num_points: int = 21,
        in_chans: int = 2,
        embed_dim: int = 768,
        norm_layer: Optional[Callable] = None,
        flatten_embedding: bool = True,
    ) -> None:
        super().__init__()

        self.num_points = num_points
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.flatten_embedding = flatten_embedding
        self.proj = nn.Linear(in_chans * num_points, embed_dim)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        B, T, N, C = x.shape
        x = x.view(B * T, N * C)  # (B * T, N * C)
        x = self.proj(x)  # (B * T, embed_dim)
        x = self.norm(x)
        if not self.flatten_embedding:
            x = x.view(B, T, -1, self.embed_dim)  # (B, T, 1, embed_dim)
        return x

    def flops(self) -> float:
        flops = self.num_points * self.in_chans * self.embed_dim
        if self.norm is not None:
            flops += self.embed_dim
        return flops


class VideoClassifierViT(nn.Module):
    """定义视频分类ViT模型结构"""
    def __init__(
        self, num_classes: int = 7, num_points: int = 21, in_chans: int = 2, embed_dim: int = 768, depth: int = 12, num_heads: int = 12, mlp_ratio: int = 4,
        qkv_bias: bool = True, qk_scale: Optional[float] = None, drop_rate: float = 0., attn_drop_rate: float = 0.0, num_frames: int = 46, dropout_rate: float = 0.3):
        super().__init__()
        self.patch_embed = PatchEmbed(num_points=num_points, in_chans=in_chans, embed_dim=embed_dim)
        self.num_patches = 1  # 每个时间步只有一个patch
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim), requires_grad=False)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.Block = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio,
                                          qkv_bias, qk_scale, drop_rate, attn_drop_rate) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(embed_dim, 2 * embed_dim)
        self.fc2 = nn.Linear(2 * embed_dim, num_classes)
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=embed_dim, num_layers=2, batch_first=True, dropout=dropout_rate)

    def forward(self, x):
        B, T, N, C = x.shape
        x = self.patch_embed(x)  # (B * T, embed_dim)
        x = x.view(B, T, -1)  # (B, T, embed_dim)
        x = x + self.pos_embed[:, 1:, :]  # (B, T, embed_dim)

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_token = cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)  # (B, T + 1, embed_dim)

        # 应用transformer blocks
        for blk in self.Block:
            x = blk(x)
        x = self.norm(x)
        x = x[:, 1:].mean(dim=1)  # (B, embed_dim)

        # LSTM处理时间维度
        lstm_out, _ = self.lstm(x.unsqueeze(1))  # (B, 1, embed_dim)
        x = lstm_out.squeeze(1)  # (B, embed_dim)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        logits_output = self.dropout(x)
        softmax_output = F.softmax(logits_output, dim=1)
        return logits_output, softmax_output


if __name__ == '__main__':
    # 测试代码
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VideoClassifierViT(num_classes=12, num_points=21, in_chans=2, embed_dim=768,
                               depth=1, num_heads=2, mlp_ratio=4, qkv_bias=True,
                               qk_scale=None, drop_rate=0.3, attn_drop_rate=0.0, num_frames=46, dropout_rate=0.3)
    model.to(device)
    x = torch.randn(32, 46, 21, 2).to(device)
    y = model(x)
    print(model)
    print(f'====model output shape: {y[1].shape}')

    # 导出onnx模型进行可视化
    from torch.onnx import export
    input = torch.randn((32,46,21,2)).to(device)
    export(model, input, "./models/Transformer_LSTM.onnx")