
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        n1 = torch.tanh(x)
        return n1
# Inputs to the model
x = torch.randn(1, 232, 116, 116)
