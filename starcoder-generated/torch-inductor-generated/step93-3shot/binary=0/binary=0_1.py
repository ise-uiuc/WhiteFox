
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1, groups=2)
        self.conv2 = torch.nn.Conv2d(2, 1, 7)
    def forward(self, x1, groups=1):
        v1 = self.conv1(x1)
        v1 = self.conv2(v1)
        if groups == 1:
            groups = torch.randn(v1.shape).long()
        v2 = v1.sum(dim=1, keepdim=True)
        return v2
# Inputs to the model
x1 = torch.randn(3,3, 64, 64)
# Inputs to the model
groups = torch.Tensor([[4]])
