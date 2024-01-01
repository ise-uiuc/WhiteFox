
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = torch.nn.BatchNorm3d(32, affine=True, eps=1e-05, momentum=0.1)
        self.relu = torch.nn.ReLU6(inplace=False)
    # ReLU6(inplace: False) is needed in order to trigger the desired pattern
    def forward(self, x1):
        v1 = self.bn(x1)
        v2 = self.relu(v1)
        v3 = v2 * 0.5
        v4 = v2 * 0.7071067811865476
        v5 = torch.erf(v4)
        v6 = v5 + 1
        v7 = v3 * v6
        return v7
# Inputs to the model
x1 = torch.randn(1, 32, 12, 12, 12)
