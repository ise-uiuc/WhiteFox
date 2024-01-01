
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias)
        v2 = v1.permute(0, 2, 1)
        v3 = v2.squeeze(0)
        softmax1 = torch.nn.functional.softmax
        v4 = softmax1(v3, dim=-1)
        return v4
# Inputs to the model
x1 = torch.randn(1, 2, 10)
