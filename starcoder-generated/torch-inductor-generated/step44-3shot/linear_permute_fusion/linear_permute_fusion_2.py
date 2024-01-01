
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 2)
    def forward(self, x0):
        v0 = x0.view(1, 3, 2)
        v1 = torch.nn.functional.linear(v0, self.linear.weight, self.linear.bias)
        v2 = v1.permute(0, 2, 1)
        return v2
# Inputs to the model
x0 = torch.randn(1, 3, 2)
