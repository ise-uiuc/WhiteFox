
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)
        self.glu = torch.nn.GLU()
        self.sigmoid = torch.nn.Linear(1, 1)
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias)
        v2 = self.glu(v1)
        v3 = v2.reshape(x1.size())
        v4 = torch.nn.functional.linear(v3, self.sigmoid.weight, self.sigmoid.bias)
        return v4
# Inputs to the model
x1 = torch.randn(2, 4, 2)
