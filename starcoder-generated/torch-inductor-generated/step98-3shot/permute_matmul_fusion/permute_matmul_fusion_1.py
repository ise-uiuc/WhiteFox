
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        return torch.pow(torch.tanh(torch.matmul(x1, torch.tanh(x2))), 2)
# Inputs to the model
x1 = torch.randn(2, 2)
x2 = torch.randn(2, 2)
