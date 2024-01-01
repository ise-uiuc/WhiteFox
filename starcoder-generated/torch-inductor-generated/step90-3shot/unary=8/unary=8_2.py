
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(32, 12, 1, stride=1, bias=True)
        self.batch_norm = torch.nn.BatchNorm2d(12, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv = torch.nn.Conv2d(12, 64, 1, stride=1, bias=True)
        self.gelu = torch.nn.GELU()
        self.conv_transpose_1 = torch.nn.ConvTranspose2d(64, 16, 1, stride=1, bias=True)
        self.conv_transpose_2 = torch.nn.ConvTranspose2d(16, 8, 1, stride=1, bias=True)
        self.dropout = torch.nn.Dropout(p=0.2)
        self.conv_transpose_3 = torch.nn.ConvTranspose2d(8, 1, 1, stride=1, bias=True)
        self.batch_norm_1 = torch.nn.BatchNorm2d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = self.batch_norm(v1)
        v3 = v2 + 3
        v4 = torch.clamp(v3, min=0)
        v5 = torch.clamp(v4, max=6)
        v6 = v2 * v5
        v7 = v6 / 6
        v8 = self.conv(v7)
        v9 = self.gelu(v8)
        v10 = torch.nn.functional.interpolate(v9, size=[768, 768], mode='nearest', align_corners=None)
        v11 = self.conv_transpose_1(v10)
        v12 = self.conv_transpose_2(v11)
        v13 = self.dropout(v12)
        v14 = self.conv_transpose_3(v13)
        v15 = self.batch_norm_1(v14)
        v16 = torch.tensor_split(v15, dim=1, indices_or_sections=[])
        return v16
# Inputs to the model
x1 = torch.randn(1, 32, 224, 224)
