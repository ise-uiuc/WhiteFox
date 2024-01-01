
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 13, 5, padding=2)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = torch.tanh(v1)
        v3 = v2[:, :, 0, 0]
        return v3
# Inputs to the model
x = torch.randn(64, 3, 56, 56)
