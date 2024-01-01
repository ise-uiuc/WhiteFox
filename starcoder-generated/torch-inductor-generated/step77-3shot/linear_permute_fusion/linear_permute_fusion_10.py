
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(in_channels=1, out_channels=2, kernel_size=2, stride=1, padding=0, bias=False)
        self.linear = torch.nn.Linear(4, 4)
    def forward(self, x1):
        x1 = self.conv(x1)
        v1 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias)
        v2 = v1.permute(0, 2, 1)
        return v2
# Inputs to the model
x1 = torch.randn(2, 1, 2, 2)
