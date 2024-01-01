
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=0, bias=False)
        self.bias = torch.nn.Parameter(data=torch.Tensor([1., 2., 3., 4., 5., 6., 7., 8.]))
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.sigmoid(v1)
        v3 = torch.mul(v1, v2)
        v4 = torch.t(v3)
        v5 = v4 + self.bias
        v6 = torch.squeeze(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 128, 128)
