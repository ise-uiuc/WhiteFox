
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)
        self.linear1 = torch.nn.Linear(4, 1)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v3 = torch.nn.functional.linear(v1, self.linear1.weight, self.linear1.bias)
        v4 = v3 + v2[:, :, :1]
        return v4
# Inputs to the model
x1 = torch.randn(1, 4, 2)
