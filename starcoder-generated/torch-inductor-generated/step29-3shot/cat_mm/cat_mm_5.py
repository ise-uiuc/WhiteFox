
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        v1 = torch.mm(x, x)
        return v1

# Inputs to the model
x = torch.randn(2, 2)
