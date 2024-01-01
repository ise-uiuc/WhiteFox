
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.avgpool = torch.nn.AvgPool2d(12, kernel_size=(12, 12), stride=(12, 12))
        self.linear = torch.nn.Linear(1440, 1000)
    def forward(self, x1):
        t1 = self.avgpool(x1)
        t2 = self.linear(t1.reshape(x1.shape[0], -1))
        t3 = 3 + t2
        t4 = torch.clamp_min(t3, 0)
        t5 = torch.clamp_max(t4, 6)
        t6 = 3 * t5
        t7 = t6 / 6
        return t7.unsqueeze(-1)
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
