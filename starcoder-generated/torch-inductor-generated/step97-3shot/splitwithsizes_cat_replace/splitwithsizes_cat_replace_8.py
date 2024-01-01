
class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = torch.nn.Conv2d(3, 3, 3, 1, 1, dilation=1, groups=1, bias=False, padding=1)
        self.conv1 = torch.nn.Conv2d(3, 16, 3, 2, 1, dilation=1, groups=1, bias=False, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 32, 3, 2, 1, dilation=1, groups=1, bias=False, padding=1)
        self.conv3 = torch.nn.Conv2d(32, 32, 3, 1, 1, dilation=1, groups=1, bias=False, padding=1)
        self.conv4 = torch.nn.Conv2d(32, 32, 3, 2, 1, dilation=1, groups=1, bias=False, padding=1)
        self.conv5 = torch.nn.Conv2d(32, 64, 3, 2, 1, dilation=1, groups=1, bias=False, padding=1)
        self.bn0 = torch.nn.BatchNorm2d(3)
    def forward(self, x):
        identity_1 = x
        out_ = self.bn0(self.conv0(x))
        out_ = (torch.add(identity_1, out_, alpha=1))
        out_ = torch.nn.AvgPool2d(3, 2, 1, count_include_pad=True)(out_)
        out_ = torch.nn.ReLU()(self.conv1(out_))
        out_ = (torch.nn.ReLU()(self.conv2(out_)))
        out_ = (torch.nn.ReLU()(self.conv3(out_)))
        out_ = (torch.nn.ReLU()(self.conv4(out_)))
        out_ = (torch.nn.ReLU()(self.conv5(out_)))
        return out_
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = Block()
    def forward(self, v1):
        split_tensors = torch.split(v1, 1, dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(v1, [1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(1, 3, 128, 128)
