
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='nearest')
        self.linear_1 = torch.nn.Linear(5, 20)
        self.linear_2 = torch.nn.Linear(20, 10)
    def forward(self, x1):
        v1 = torch.relu(x1)
        v2 = torch.cat([x1, 2 * v1], 1)
        v3 = self.linear_1(v2)
        v4 = self.linear_2(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 5)
