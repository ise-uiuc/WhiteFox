
class Layer1(torch.nn.Module):
    def __init__(self, inp, hidden, out):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 3, 1, 1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(32, affine=False, track_running_stats=False)
    def forward(self, v1):
        v1_branch_1 = torch.nn.ReLU()(self.bn1(self.conv1(v1)))
        return v1_branch_1
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.branch_1 = Layer1(3, 16, 32)
        self.conv1 = torch.nn.Conv2d(3, 32, 3, 1, 1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(32, affine=False, track_running_stats=False)
        self.conv2 = torch.nn.Conv2d(32, 32, 3, 1, 1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(32, affine=False, track_running_stats=False)
    def forward(self, v1):
        v1_branch_1 = torch.nn.ReLU()(self.bn1(self.conv1(v1)))
        v1_branch_2 = torch.nn.ReLU()(self.bn2(self.conv2(self.branch_1(v1))))
        return v1_branch_2
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
