
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn_1= torch.nn.BatchNorm2d(32, eps=1e-04, momentum=0, affine=False)
    def forward(self, x):
        v1 = self.bn_1(x)
        v2 = torch.tanh(v1)
        return v2
# Inputs to the model
x = torch.randn(2, 32, 64, 64)
