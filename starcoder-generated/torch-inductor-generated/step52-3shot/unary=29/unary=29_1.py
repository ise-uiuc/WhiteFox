
class Model(torch.nn.Module):
    def __init__(self, min_value=-0.1332, max_value=-1.1488):
        super().__init__()
        self.relu = torch.nn.ReLU(inplace=False)
        self.batch_normalization = torch.nn.BatchNorm2d(47, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv_transpose2d_1 = torch.nn.ConvTranspose2d(47, 16, (1, 5), stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1)
        self.conv_transpose2d_2 = torch.nn.ConvTranspose2d(16, 2, 5, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x8):
        v5 = self.batch_normalization(x8)
        v8 = self.relu(v5)
        v10 = self.conv_transpose2d_1(v8)
        v11 = self.relu(v10)
        v12 = self.conv_transpose2d_2(v11)
        v13 = torch.clamp_min(v12, self.min_value)
        v14 = torch.clamp_max(v13, self.max_value)
        return v14
# Inputs to the model
x8 = torch.randn(1, 47, 28, 8)
