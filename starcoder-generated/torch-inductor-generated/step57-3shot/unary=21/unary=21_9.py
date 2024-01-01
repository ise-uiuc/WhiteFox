
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        v1 = x
        v2 = torch.tanh(v1)
        return v2
# Inputs to the model
x = torch.randn(1, 1, 256, 256)
