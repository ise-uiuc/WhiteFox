
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.avgpool = torch.nn.AvgPool2d(kernel_size=(5, 5), stride=1, padding=0)
    def forward(self, x1):
        v1 = self.avgpool(x1)
        v2 = v1.reshape(16, 256, 1, 1)
        v3 = F.sigmoid(v2)
        v4 = v2 * v3
        v5 = x1 * v4
        return v5
# Inputs to the model
x1 = torch.randn(16, 256, 64, 64)
