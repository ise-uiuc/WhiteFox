
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.utils.spectral_norm(torch.nn.Conv2d(64, 128, 3, stride=2, padding=1, padding_mode='replicate'))
        self.conv2 = torch.nn.Conv2d(128, 256, 7, stride=1, padding=3, dilation=2, groups=2)
        self.conv3 = torch.nn.Linear(1152, 3)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.nn.functional.group_norm(v1, 2)
        v3 = torch.relu(v1)
        v4 = self.conv2(v3)
        v5 = torch.nn.functional.instance_norm(v4, affine=False)
        v6 = torch.relu(v5)
        v7 = v7 = v5.reshape(v4.numel()//2, 16, 256, 32)
        v8 = v6.reshape(v4.numel()//16, 256, 32, 3)
        v9 = v8.sum()
        return v9
# Inputs to the model
x1 = torch.randn(1, 64, 224, 224)
