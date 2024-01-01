
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.view = torch.reshape
    def forward(self, x1):
        v1 = x1.detach()
        v2 = self.view(v1, (1, 4))
        v3 = torch.nn.functional.linear(v2, self.linear.weight, self.linear.bias)
        return v3
# Inputs to the model
x1 = torch.randn(2, 2)
