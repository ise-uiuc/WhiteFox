
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, dim):
        return torch.cat([x, x+1, x+2, x+3], dim)
# Inputs to the model
x = torch.randn(1, 1)
dim = 1
