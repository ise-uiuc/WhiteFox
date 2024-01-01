
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(6, 2)
        self.conv = torch.nn.Conv2d(3, 1, 1)
    def forward(self, x1):
        v1 = x1.transpose(1, 2)
        v2 = torch.sum(v1, dim=0, keepdim=True)
        v3 = x1.permute(0, 2, 1)
        v4 = torch.nn.functional.linear(v3, self.linear.weight, self.linear.bias)
        return v4
# Inputs to the model
x1 = torch.randn(1, 2, 6)
