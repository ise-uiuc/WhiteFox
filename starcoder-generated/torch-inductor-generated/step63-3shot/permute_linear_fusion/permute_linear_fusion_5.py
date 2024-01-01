
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2, bias=False)
        self.flatten = torch.nn.Flatten(0, 1)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, bias=None)
        x2 = self.flatten(v2)
        return x2
# Inputs to the model
x1 = torch.randn(1, 2, 2)
