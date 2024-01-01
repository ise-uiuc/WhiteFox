
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        v1 = torch.tanh(x)
        return v1
# Inputs to the model
x = torch.randn(1, 8, 168, 626)
