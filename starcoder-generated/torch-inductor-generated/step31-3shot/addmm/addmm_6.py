
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        v1 = torch.mm(x, x)
        v2 = v1 + x
        return v2
# Input to the model
x = torch.randn(3, 3)
