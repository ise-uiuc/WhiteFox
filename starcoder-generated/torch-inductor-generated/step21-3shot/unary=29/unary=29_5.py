
class Model(torch.nn.Module):
    def __init__(self, min_value=1.1, max_value=4):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 3, stride=1)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.ConvTranspose2d(16, 8, 3, stride=1)
        self.max_value = max_value
        self.min_value = min_value
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.relu(v1)
        v3 = self.conv2(v2)
        v4 = torch.clamp_min(v3, self.min_value)
        v5 = torch.clamp_max(v4, self.max_value)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 56, 56)
