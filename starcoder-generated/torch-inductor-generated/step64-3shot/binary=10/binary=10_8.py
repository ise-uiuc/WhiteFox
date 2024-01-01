
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64, 16, bias=False)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + o2
        return v2
o2 = torch.randn(16, 16, requires_grad=True)
m = Model()
x1 = torch.randn(1, 64)
