
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(32, 3, 1, stride=1)
    def forward(self, torch.randn(10, 32, 224, 224)):
        v = self.conv(torch.randn(10, 32, 224, 224))
        return torch.tanh(v)
# Inputs to the model
torch.randn(10, 32, 224, 224)
