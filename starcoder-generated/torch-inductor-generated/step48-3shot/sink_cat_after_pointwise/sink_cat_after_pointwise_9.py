
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.sin(2 * x).sum(2) + torch.tanh(3 * x).max(1)[0]
# Inputs to the model
x = torch.randn(2, 3, 4)
