
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2, bias=False)
    def forward(self, x1, x2):
        v1 = torch.nn.functional.linear(x1, self.linear.weight)
        v2 = v1.permute(0, 1, 3, 2)
        v3 = torch.nn.functional.linear(x1 - v1, self.linear.weight)
        return (v2, v3)
# Inputs to the model
x1 = torch.randn(1, 2, 2, 2)
x2 = torch.randn(1, 2, 2, 2)
