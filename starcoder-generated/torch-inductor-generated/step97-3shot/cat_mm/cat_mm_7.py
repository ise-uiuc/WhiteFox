
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        return torch.cat([torch.mm(x1, x1) for i in range(7)], 0) # A list with 7 copies of the same tensor
# Inputs to the model
x1 = torch.randn(32, 32)
