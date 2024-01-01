
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 2)
    def forward(self, x1, x2):
        t = torch.cat([x1, x2], dim=1)
        v1 = torch.nn.functional.linear(t, self.linear.weight, self.linear.bias)
        k = v1.permute(0, 2, 1)
        return k
# Inputs to the model
x1 = torch.randn(3, 1)
x2 = torch.randn(3, 1)
