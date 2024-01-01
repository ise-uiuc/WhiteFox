
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 8, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(1, 8, 1, stride=1, padding=0)
    def forward(self, x1):
        # Input: [N, C, H, W] (input_tensor) [1, 1, 64, 64]
        v1 = self.conv1(x1) # [1, 8, 64, 64]
        v2 = torch.relu(v1) # [1, 8, 64, 64]
        v3 = self.conv2(v2) # [1, 8, 64, 64]
        v4 = torch.relu(v3) # [1, 8, 64, 64]
        return v4 # [1, 8, 64, 64]
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
