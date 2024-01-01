
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 32, 7, stride=1, padding=3)
        self.pool0 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        self.fc1 = torch.nn.Linear(8, 8)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 - 5
        v3 = F.relu(v2)
        v4 = self.pool0(v3)
        v5 = v4.flatten(start_dim=1)
        v6 = self.fc1(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 1, 28, 28)
