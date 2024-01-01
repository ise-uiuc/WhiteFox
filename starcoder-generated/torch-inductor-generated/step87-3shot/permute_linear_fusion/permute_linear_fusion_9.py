
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v1 = v1.permute(0, 2, 1)
        v1 = v2.clamp(min=0, max=0.5)
        v3 = v1 ** 0.5
        v4 = torch.sum(torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias), dim=[1,2]) / 3
        return (v3, v4, v2, v3)
# Inputs to the model
x1 = torch.randn(1, 3, 3)
