
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v3 = v2.tanh()
        y2 = v3.clone()
        y2 = y2 - 2.0
        return y2
# Inputs to the model
x1 = torch.randn(1, 2, 2)
