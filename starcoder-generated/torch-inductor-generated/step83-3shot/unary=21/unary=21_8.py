
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = torch.nn.MaxPool2d(3, stride=1, padding=1)
    def forward(self, x):
        v1 = self.pool(x)
        v2 = torch.tanh(v1)
        return v2
# Inputs to the model
x = torch.randn(1, 3, 225, 225)
