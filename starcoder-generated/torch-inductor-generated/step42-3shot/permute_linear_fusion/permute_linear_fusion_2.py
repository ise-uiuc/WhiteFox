
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.maxpool = torch.nn.MaxPool1d(2, stride=1, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        self.mean = torch.mean
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v3 = self.maxpool(v2)
        v4 = self.mean(v3, dim=[-2, -1])
        return v4
# Inputs to the model
x1 = torch.randn(1, 2, 2)
