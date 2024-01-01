
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, kernel_size=3, padding=2)
    def forward(self, x):
        x1 = self.conv1(x)
        x1 = torch.cat([x, x1], axis=1)
        x2 = torch.tanh(x1)
        x2 = torch.cat([x1, x2], axis=1)
        return x2
# Inputs to the model
x = torch.randn(1, 3, 35, 35)
