
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(3, 2, 1, 1)
    def forward(self, x1):
        x1 = x1 + 100.1
        x2 = torch.nn.functional.relu(x1) - 1000.1
        x3 = x2.permute(1, 0, 2, 3)
        x4 = torch.nn.functional.linear(x3.flatten(1), torch.ones(8, 2)) - 5.5395
        x4 = torch.nn.functional.relu(x4) * 8.75
        return torch.mean(x4, dim=0)
# Inputs to the model
x1 = torch.randn(2, 3, 2, 2)
