
class ModelRelu(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 2, 1)
    def forward(self, x2):
        v1 = torch.relu(self.conv(x2))
        return v1
# Inputs to the model
x2 = torch.randn(1, 3, 48, 48)
