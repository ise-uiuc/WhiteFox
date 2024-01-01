
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.conv = torch.nn.Conv2d(2, 2, 3, bias=False)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        x2 = self.conv(v2)
        x3 = torch.nn.functional.relu(x2) + x2
        x4 = torch.nn.functional.relu(x3.detach())
        x5 = torch.nn.functional.max_pool2d(x4)
        x5 = x5.permute(0, 2, 1)
        x6 = torch.nn.functional.linear(x5, self.linear.weight, self.linear.bias)
        x7 = x2 + x6
        x8 = torch.nn.functional.relu(x6)
        x9 = torch.nn.functional.relu(x7)
        return torch.randn(x9.shape[0], x9.shape[1], x9.shape[2], x9.shape[3])
# Inputs to the model
x1 = torch.randn(1, 2, 2)
