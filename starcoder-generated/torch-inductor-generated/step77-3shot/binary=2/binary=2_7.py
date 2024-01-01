
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        conv1_weight = torch.randn(61, 70, 1, 3, requires_grad=False)
        groups = 1
        self.conv1 = torch.nn.Conv2d(915, 35, 3, stride=1, padding=1, dilation=2, groups=groups, bias=False)
        conv3_weight = torch.randn(35, 13, 23, 7, requires_grad=True)
        self.conv3 = torch.nn.Conv2d(35, 885, 3, stride=1, padding=3, dilation=1, groups=groups, bias=False)
        conv2_weight = torch.randn(26, 77, 3, 1, requires_grad=False)
        self.conv2 = torch.nn.Conv2d(77, 265, 3, stride=1, padding=2, dilation=4, groups=groups, bias=True)
        conv4_weight = torch.randn(78, 19, 3, 1, requires_grad=False)
        self.conv4 = torch.nn.Conv2d(543, 265, 1, stride=1, padding=1, dilation=1, groups=groups, bias=True)
        self.conv = torch.nn.Conv2d(265, 265, 1, stride=2, padding=3, dilation=7, groups=groups, bias=False)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = torch.nn.functional.dropout(v1, p=0.5)
        v3 = v2.view((-1,35))
        v3 = torch.matmul(v3, conv2_weight)
        v4 = torch.transpose(conv3_weight, 1, 0)
        v5 = v4.detach()
        v4 = v5 + v3
        v6 = torch.transpose(conv4_weight, 1, 0)
        v7 = v6.detach()
        v6 = v7 * v2
        v8 = v3 + v6
        v4 = v3.size()
        x = v8.view(math.floor_divide(v4[0], 543), 265, 142, 85)
        v1 = self.conv2(x)
        v4 = v1.size()
        v6 = math.floor_divide(v4[0], 885)
        x = self.conv3(v1)
        v4 = x.size()
        x = x.view(v6, 885, 287, 130)
        x = self.conv4(x)
        v4 = x.size()
        x = x.view(v6, 543, 88)
        x = x + v8
        x = x.view(v6, 265*88)
        x = self.conv(x)
        v4 = x.size()
        x = x.view((-1,265,37))
        x = x - v1
        return x
# Inputs to the model
x = torch.randn(1, 915, 191, 221)
