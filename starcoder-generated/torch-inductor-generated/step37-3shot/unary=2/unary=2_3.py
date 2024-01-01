
class Module0(torch.nn.Module):
    def __init__(self, dim0):
        super().__init__()
        self.conv_transpose3d = torch.nn.ConvTranspose3d(128, 32, (3, 3, 3), (1, 1, 1), (1, 1, 1), (1, 1, 1))
        self.conv3dtranspose3d = torch.nn.Conv3dTranspose(1, 2, 4, 2, 1, 2, 1)
    def forward(self, x, x3):
        v37, v28 = self.conv_transpose3d.forward(x)
        v3, v2, v1 = x3.size()
        v32, v23 = self.conv3dtranspose3d.forward(v37.view(-1, v2, v3, v1))
        ret0 = v32.view(-1, v32.size()[2], v32.size()[3], v32.size()[4])
        return ret0
class Module1(torch.nn.Module):
    def __init__(self, dim0):
        super().__init__()
        self.module0_0 = Module0(dim0)
        self.conv_transpose3d = torch.nn.ConvTranspose3d(32, 2, (3, 3, 3), (1, 1, 1), (1, 1, 1), (1, 1, 1))
    def forward(self, x0):
        v4 = self.module0_0.forward(x0, x0)
        v43, v14 = self.conv_transpose3d.forward(v4)
        ret1 = v43
        return ret1
class Model(torch.nn.Module):
    def __init__(self, dim0):
        super().__init__()
        self.module1_0 = Module1(dim0)
        self.conv_transpose2d = torch.nn.ConvTranspose2d(14, 1, 4, stride=2)
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        v34 = self.module1_0.forward(x)
        v11, v12 = v34.size()
        v18 = v34.view(-1, v11, v12)
        v13 = self.conv_transpose2d(v18)
        v19 = self.relu(v13)
        return v19
# Inputs to the model
x = torch.randn(1, 3, 224, 224)
