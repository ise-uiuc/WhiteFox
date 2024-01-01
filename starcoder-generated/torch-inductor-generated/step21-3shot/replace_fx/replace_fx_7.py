
class ModelNew2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 2, 2)
        self.batchnorm = torch.nn.BatchNorm2d(2)

    def forward(self, x0_1):
        x1_1 = self.conv(x0_1)
        x2_1 = self.batchnorm(x1_1)
        x3_1 = torch.rand_like(x1_1)
        x4_1 = torch.nn.functional.dropout(x3_1)
        x5_1 = torch.nn.functional.dropout(x4_1)
        x6_1 = torch.nn.functional.dropout(x5_1)
        return x6_1
# Inputs to the model
x0_1 = torch.randn(2, 2, 2, 2)
