
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(512, 2688, 1, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv3 = torch.nn.Conv2d(2688, 2688, 3, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv4 = torch.nn.Conv2d(2688, 2688, 1, stride=1, padding=0, dilation=1, groups=1, bias=True)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v1_reshape = torch.reshape(v1, (-1, 2688, 1))
        v1_reshape_T = torch.transpose(v1_reshape, 1, 2)
        v2 = self.conv3(v1_reshape_T)
        v3 = torch.transpose(v1_reshape_T, 1, 2)
        v3_reshape = torch.reshape(v3, (-1, 2688, 1))
        v4 = self.conv4(v3_reshape)
        v4_tile = torch.transpose(v4, 1, 2)
        return v4_tile
# Input to the model
x1 = torch.randn(1, 512, 7, 7)
