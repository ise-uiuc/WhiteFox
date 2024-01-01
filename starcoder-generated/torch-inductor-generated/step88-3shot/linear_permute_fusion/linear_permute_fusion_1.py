
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 2)
    def forward(self, x5):
        v5 = torch.nn.functional.linear(x5, self.linear.weight, self.linear.bias)
        v6 = v5.permute(0, 2, 1)
        return v5
# Inputs to the model
x5 = torch.randn(2, 2, 3)
