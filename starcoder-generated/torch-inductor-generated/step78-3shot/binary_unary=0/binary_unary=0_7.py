
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_16 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.linear_1 = torch.nn.Linear(56 * 56 * 16, 4096, bias=True)
        self.linear_2 = torch.nn.Linear(4096, 256, bias=True)
        self.linear_3 = torch.nn.Linear(256, 10, bias=True)
    def forward(self, x):
        v1 = self.conv_16(x)
        v2 = v1.view(-1, 56 * 56 * 16)
        v3 = self.linear_1(v2)
        v4 = self.linear_2(v3)
        v5 = self.linear_3(v4)
        v6 = v5 + v5
        v7 = torch.relu(v6)
        return v7
# Inputs to the model
x = torch.randn(1, 16, 64, 64)
