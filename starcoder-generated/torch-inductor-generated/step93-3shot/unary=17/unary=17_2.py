
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_1 = torch.nn.Linear(5, 2)
        self.conv_1 = torch.nn.Conv2d(1, 1, 2)
        self.linear_1_1 = torch.nn.Linear(1, 5)
    def forward(self, x1):
        v1 = self.linear_1(x1)
        v2 = v1.reshape(1, 1, 2, 2)
        v3 = self.conv_1(v2)
        v4 = F.relu(v3)
        v5 = self.linear_1_1(v4)
        v6 = v5.reshape(1, 16)
        return v6
# Inputs to the model
x1 = torch.randn(3, 5)
