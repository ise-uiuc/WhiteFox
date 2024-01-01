
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.softmax = torch.nn.Softmax()
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v5 = self.softmax(v1)
        v3 = torch.max(v2, dim=2, keepdim=True)[0]
        return v3
# Inputs to the model
x1 = torch.randn(1, 2, 2)
