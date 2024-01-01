
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(2, 2, 2)
        self.conv2 = torch.nn.Conv2d(2, 2, 2)
        self.bn = torch.nn.BatchNorm2d(2, affine=True)
    def forward(self, x7):
        conv1_res = self.conv1(x7)
        conv2_res = self.conv2(conv1_res)
        bn_res = self.bn(conv2_res)
        return bn_res
# Inputs to the model
x7 = torch.randn(1, 2, 8, 8)
