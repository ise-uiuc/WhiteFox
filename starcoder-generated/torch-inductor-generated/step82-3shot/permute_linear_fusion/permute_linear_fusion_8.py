
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.softmax = torch.nn.Softmax()
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v3 = v1.squeeze(1)
        v4 = v1.squeeze(2)
        v5 = torch.nn.functional.softmax(torch.stack([v2, v3, v4], dim=0), dim=0)
        return v5
# Inputs to the model
x1 = torch.randn(1, 2, 2)
