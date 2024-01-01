
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.rand_like(x)
# Inputs to the model
x = torch.randn(1, 2, 2)
