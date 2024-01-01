
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.softmax = torch.nn.Softmax()
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v3 = v2.unsqueeze(-1)
        v4 = v3.squeeze(-1)
        v3 = self.softmax(v4)
        return v3
# Inputs to the model
x1 = torch.randn(1, 2, 2)
