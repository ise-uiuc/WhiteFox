
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tanh = torch.nn.Tanh()
        self.conv = torch.nn.Conv2d(64, 64, 3, padding=1)
        self.conv1 = torch.nn.Conv2d(64, 64, 3, padding=1)
    def forward(self, x):
        n1 = self.conv(x)
        n2 = self.conv1(n1)
        n3 = self.tanh(n2)
        return n3
# Inputs to the model
tensor = torch.randn(1, 64, 128, 128)
