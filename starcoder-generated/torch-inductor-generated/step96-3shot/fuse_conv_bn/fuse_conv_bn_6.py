
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(113)
        self.layer1 = torch.nn.Conv2d(6, 6, 1)
        torch.manual_seed(11)
        self.layer2 = torch.nn.BatchNorm2d(6)
    def forward(self, x2):
        y3 = torch.nn.functional.conv2d(x2, self.layer1.weight, self.layer1.bias, self.layer1.stride, self.layer1.padding, self.layer1.dilation, self.layer1.groups)
        y3 = torch.nn.functional.batch_norm(y3, self.layer2.running_mean, self.layer2.running_var, self.layer2.weight, self.layer2.bias, self.layer2.training, self.layer2.momentum, self.layer2.eps)
        x2 = y3 + y3
# Inputs to the model
x3 = torch.randn(1, 6, 6, 6)
