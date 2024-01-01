
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 3, bias=False)
        self.conv = torch.nn.Conv2d(3, 3, 2, bias=True)
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, self.linear.weight.T)
        v2 = self.conv(v1)
        v3 = v2.permute(0, 1, 3, 2)
        return v3
# Inputs to the model
x1 = torch.randn(3, 2)
