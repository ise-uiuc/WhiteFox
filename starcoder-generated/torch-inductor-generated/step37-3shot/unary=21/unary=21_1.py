
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        v2 = torch.tanh(x)
        return v2
# Inputs to the model
x = torch.randn(1, 15, 25)
