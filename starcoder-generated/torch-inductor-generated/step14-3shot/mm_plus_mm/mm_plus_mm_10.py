
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer1 = torch.nn.Linear(10, 10)
    def forward(self, x1, x2, x3, x4):
        v1 = torch.mm(x1, x2)
        v2 = torch.mm(x3, x4)
        v3 = v1 + self.linear_layer1(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 10)
x2 = torch.randn(1, 10)
x3 = torch.randn(1, 10)
x4 = torch.randn(1, 10)
