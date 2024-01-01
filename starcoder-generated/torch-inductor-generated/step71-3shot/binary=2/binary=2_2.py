
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.linear_1 = torch.nn.Linear(2048, 1000)
        self.linear_2 = torch.nn.Linear(1000, 1000)
        self.linear_3 = torch.nn.Linear(1000, 133)
    def forward(self, x):
        v1 = self.avg_pool(x)
        v2 = torch.flatten(v1, 1)
        v3 = self.linear_1(v2)
        v4 = self.linear_2(v3)
        v5 = self.linear_3(v4)
        v6 = v5 - 45.1
        return v6
# Inputs to the model
x = torch.randn(1, 3, 224, 224)
