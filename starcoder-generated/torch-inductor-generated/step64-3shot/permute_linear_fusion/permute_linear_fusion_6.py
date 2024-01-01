
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.tanh = torch.nn.Tanh()
        self.linear1 = torch.nn.Linear(2, 2)
        self.gelu = torch.nn.GELU()
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v3 = self.tanh(v2)
        v4 = torch.nn.functional.linear(v1, self.linear1.weight, self.linear1.bias)
        del v1
        z2 = v3 + v4
        return self.gelu(z2)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
