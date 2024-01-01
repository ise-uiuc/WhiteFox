
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(5, 4, bias=False)
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, self.linear.weight, None)
        return v1
# Inputs to the model
x1 = torch.randn(1, 2, 5)
