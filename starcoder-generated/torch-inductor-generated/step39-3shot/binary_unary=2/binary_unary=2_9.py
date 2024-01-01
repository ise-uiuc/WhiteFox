
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 11, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 - torch.zeros(v1.shape, device=v1.device) # Use zero tensor to perform the subtraction
        v3 = F.relu(v2)
        v4 = v3 - torch.tensor(0.5, dtype=torch.float32, device=v3.device) # Use a float scalar to perform the subtraction
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
