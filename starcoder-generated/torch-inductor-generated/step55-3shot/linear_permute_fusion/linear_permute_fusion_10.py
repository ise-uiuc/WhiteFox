
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn1  = torch.nn.BatchNorm2d(64)
        self.conv = torch.nn.ConvTranspose2d(1, 1,
                                              kernel_size=(3,3),
                                      stride=(1,1),
                                          bias=False)
    def forward(self, x):
        v7 = torch.nn.functional.conv_transpose2d(x, self.conv.weight, bias=None)
        v1 = self.bn1(v7)
        v2 = v1.permute(0, 2, 1)
        return v2.flip(0)
# Inputs to the model
x = torch.randn(1, 1, 3, 3).cuda()
