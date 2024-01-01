
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v3 = torch.log(v2)
        v4 = torch.max(v3, dim=1)[0]
        v5 = torch.mean(v3)
        return v4, v5
# Inputs to the model
x1 = torch.randn(1, 2, 2)
