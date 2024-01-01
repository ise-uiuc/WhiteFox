
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(480, 640)
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias).permute(0, 2, 1)
        v2 = v1.permute(0, 2, 1)
        v3 = torch.ops.aten.addmm(v1, v2, self.linear.weight)
        return torch.ops.aten.bmm(v3, v1)
# Inputs to the model
x1 = torch.randn(1, 100, 2, 2)
