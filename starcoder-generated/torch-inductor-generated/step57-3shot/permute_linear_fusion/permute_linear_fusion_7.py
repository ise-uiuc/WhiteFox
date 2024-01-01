
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear0 = torch.nn.Linear(2, 2)
        self.tanh = torch.nn.Tanh()
        self.linear1 = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear0.weight, self.linear0.bias)
        v3 = self.tanh(v2)
        v4 = torch.nn.functional.linear(v3, self.linear1.weight, self.linear1.bias)
        return v4
# Inputs to the model
x1 = torch.randn(2, 2, 2)
