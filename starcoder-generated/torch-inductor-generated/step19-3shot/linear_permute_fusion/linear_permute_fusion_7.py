
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1, x2):
        v1 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias)
        v2 = torch.tanh(v1)
        v2 = v1 + v2
        return v2
# Inputs to the model
x1 = torch.randn(2, 2, 2, device="cpu")
x2 = torch.randn(2, 2, 2, device="cpu")
