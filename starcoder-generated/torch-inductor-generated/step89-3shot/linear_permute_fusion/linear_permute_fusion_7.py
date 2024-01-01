
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias)
        v2 = v1.permute(1, 2, 0)
        v3 = v2.transpose(0, 2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 2, 2)
