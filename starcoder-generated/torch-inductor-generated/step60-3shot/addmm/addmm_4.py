
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.t1 = torch.randn(3, 3)
        self.t2 = torch.randn(3, 3)
    def forward(self, x):
        v1 = torch.mm(x, x)
        v2 = v1 + x
        return v2
# Inputs to the model
x = torch.randn(3, 3)
