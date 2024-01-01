
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v = torch.mm(x1, x1)
        v = torch.cat([v, v, v, v], 0) # <-- additional concat here
        return v
# Inputs to the model
x1 = torch.randn(2, 2)
