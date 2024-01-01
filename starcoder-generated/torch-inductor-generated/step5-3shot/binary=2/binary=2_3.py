
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.tanh(x) + torch.sigmoid(x)*5
# Inputs to the model
x = torch.randn(1, 3, 64, 64)
