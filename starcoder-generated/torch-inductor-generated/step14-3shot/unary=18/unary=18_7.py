
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.batch_norm1 = torch.nn.BatchNorm2d(num_features=64, eps=0.05, momentum=0.05)
        self.batch_norm2 = torch.nn.BatchNorm2d(num_features=32, eps=0.01, momentum=0.01)
        self.avg_pool = torch.nn.AdaptiveAvgPool2d((7, 7))
    def forward(self, x7):
        v1 = self.batch_norm1(x7)
        v2 = torch.sigmoid(v1)
        v3 = self.batch_norm2(v2)
        v4 = torch.sigmoid(v3)
        v5 = self.avg_pool(v4)
        return v5
# Inputs to the model
x7 = torch.randn(32, 32, 7, 7)
