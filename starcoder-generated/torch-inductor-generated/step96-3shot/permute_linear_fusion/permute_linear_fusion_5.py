
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.conv = torch.nn.Conv2d(1, 3, 1)
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1.permute(0, 2, 1)
        v3 = torch.nn.functional.linear(v2, self.linear.weight, self.linear.bias)
        v4 = self.conv(torch.unsqueeze(v3, 1)).squeeze()
        return torch.relu(v4)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
