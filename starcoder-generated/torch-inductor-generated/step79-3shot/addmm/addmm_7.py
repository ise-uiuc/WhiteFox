
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.t1 = torch.randn(3, 3)
    def forward(self, x1):
        v1 = torch.mm(x1, self.t1)
        v2 = x1 + self.t1
        v3 = torch.mm(v2, v2)
        return v1 + v3
# Inputs to the model
x1 = torch.randn(3, 3)
