
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 5)
    def forward(self, x1):
        v0 = x1
        v1 = torch.nn.functional.linear(v0, self.linear.weight, self.linear.bias)
        v2 = v1.permute(0, 2, 1)
        return v2.contiguous().permute(-2, -1, -3).squeeze()

# Inputs to the model
x1 = torch.randn(1, 3, 4)

