
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x1):
        z = self.linear.weight
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        x2 = self.sigmoid(v2)
        y = torch.rand(1, 2, 3, requires_grad=True)
        a = torch.nn.functional.conv1d(x2, y, None, 1, (0, 0), 1, 1, False, [], False)
        return a
# Inputs to the model
x1 = torch.randn(1, 2, 2)
