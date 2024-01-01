
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.threshold = torch.nn.Threshold(-0.128962087, -0.128962087, 5.51179533e-07)
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias)
        v2 = v1.sum(dim=-1)
        v3 = self.threshold(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 2, 2)
