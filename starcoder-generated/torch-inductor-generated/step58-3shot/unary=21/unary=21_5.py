
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 4, 3)
        self.tanh = torch.nn.Tanh()
    def forward(self, x):
        x = self.conv(x)
        x = torch.tanh(x.unsqueeze(2)).squeeze(2)
        x = torch.tanh(x.unsqueeze(2)).squeeze(2)
        return x
# Inputs to the model
x = torch.randn(1, 4, 64, 64)
