
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 2)
        self.linear2 = torch.nn.Linear(2, 2)
        self.conv = torch.nn.Conv2d(1, 1, (1, 1), stride=(2, 2))
    def forward(self, x1):
        x2 = torch.nn.functional.linear(x1, self.linear1.weight, self.linear1.bias)
        x3 = torch.nn.functional.linear(x2, self.linear2.weight, self.linear2.bias)
        x4 = x3.permute(2, 1, 0).unsqueeze(0)
        x5 = self.conv(x4)
        return x5
# Inputs to the model
x1 = torch.randn(2, 2)
