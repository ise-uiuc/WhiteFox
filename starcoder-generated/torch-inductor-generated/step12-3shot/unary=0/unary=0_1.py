
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 1, 7, stride=6, padding=4, dilation=2)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 * torch.tanh(torch.randn(1))
        v3 = v1 * 0.044715
        v4 = v2 + v3
        v5 = torch.nn.functional.relu(v4)
        v6 = v1 + v5
        return v6
# Inputs to the model
x = torch.randn(3, 3, 10, 10)
