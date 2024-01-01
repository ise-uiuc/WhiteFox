
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.Conv2d_1_3x3 = nn.Conv2d(1, 3, kernel_size=(3, 3), stride=(1, 1), bias=False)
        self.Conv2d_2_3x3 = nn.Conv2d(3, 1, kernel_size=(3, 3), stride=(1, 1), bias=False)
        self.BatchNorm2d_1 = nn.BatchNorm2d(3, eps=0.0010000000474974513, momentum=0.0, affine=True, track_running_stats=True)
        self.BatchNorm2d_2 = nn.BatchNorm2d(1, eps=1e-05, momentum=0.0, affine=True, track_running_stats=True)
        self.ReLU_1 = nn.ReLU()
        self.ReLU_2 = nn.ReLU()
    def forward(self, x):
        out = self.Conv2d_1_3x3(x)
        out = self.BatchNorm2d_1(out)
        out = self.ReLU_1(out)
        out = self.Conv2d_2_3x3(out)
        out = self.BatchNorm2d_2(out)
        out = self.ReLU_2(out)
        return out
x = torch.randn(1, 1, 224, 224)
