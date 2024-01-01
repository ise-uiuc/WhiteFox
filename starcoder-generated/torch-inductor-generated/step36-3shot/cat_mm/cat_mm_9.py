
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        v = torch.rand(16, 28, 28, 56)
        return torch.cat(v, 2)
# Inputs to the model
x = torch.rand(1, 32*32*16)
