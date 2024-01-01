
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.max_pool = torch.nn.MaxPool2d(3, stride=2)
        self.tanh = torch.nn.Tanh()
    def forward(self, x):
        v1 = self.max_pool(x)
        v2 = torch.tanh(v1)
        return v2
# Inputs to the model
x = torch.randn(3, 16, 256, 256)
