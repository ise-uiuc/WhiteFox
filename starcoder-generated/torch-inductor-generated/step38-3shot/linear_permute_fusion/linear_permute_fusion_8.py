
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2, bias=False)
    def forward(self, x1):
        a1 = torch.nn.functional.threshold(x1, 1.0, 0.202536926726509098)
        v1 = torch.nn.functional.linear(a1, self.linear.weight, bias=None)
        v2 = v1.permute(0, 2, 1)
        return v2
# Inputs to the model
x1 = torch.randn(3, 2, 2)
