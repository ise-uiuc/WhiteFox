
class Model(torch.nn.Module):
    def __init__(self, min_value=1.829, max_value=2.87):
        super().__init__()
        self.conv_transpose1 = torch.nn.ConvTranspose2d(3, 16, 1, stride=1, padding=1)
        self.bn_fold1 = torch.nn.BatchNorm2d(32, eps=1e-005, momentum=0.1, affine=True, track_running_stats=True)
        self.relu1 = torch.nn.ReLU()
        self.conv_transpose2 = torch.nn.ConvTranspose2d(32, 3, 1, stride=1, padding=1)
        self.bn_fold2 = torch.nn.BatchNorm2d(3, eps=1e-005, momentum=0.1, affine=True, track_running_stats=True)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.conv_transpose1(x1)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        v4 = torch.nn.InstanceNorm2d(32, affine=True, track_running_stats=True)(v3)
        v5 = torch.nn.ReLU(inplace=False)(v4)
        v6 = self.conv_transpose2(v5)
        v7 = torch.clamp_min(v6, self.min_value)
        v8 = torch.clamp_max(v7, self.max_value)
        v9 = self.bn_fold2(v8)
        return v9
# Inputs to the model
x1 = torch.randn(1, 3, 128, 128)
