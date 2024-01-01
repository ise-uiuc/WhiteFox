
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 3)
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias)
        v2 = v1.permute(0, 2, 1)
        softmax1 = torch.nn.functional.softmax
        v3 = softmax1(v2, dim=-1)
        return v3
# Inputs to the model
x1 = torch.randn(1, 2, 2)
