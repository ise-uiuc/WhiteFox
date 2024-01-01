
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 60)
    def forward(self, x1):
        y = torch.randn(1, 60, 2)
        b = torch.randn(1, 2, 3)
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, b, c)
        return v2
# Inputs to the model
x1 = torch.randn(1, 2, 3)
