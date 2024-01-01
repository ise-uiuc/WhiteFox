
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(13)
        # TODO: Add `bias=False` argument to torch.nn.Conv2d
        self.conv1 = torch.nn.Conv2d(1, 1, 1, stride=1, padding=0)
        # TODO: Add `track_running_stats=False` argument to `torch.nn.BatchNorm2d`
        self.bn = torch.nn.BatchNorm2d(2)
        # TODO: Use the functional version of the torch.nn.Conv2d module
        self.conv2 = torch.nn.Conv2d(2, 2, 2, stride=2, padding=0)
    def forward(self, x2):
        y2 = self.conv1(x2)
        y2 = self.bn(y2[0])
        # TODO: Use the functional version of the torch.nn.BatchNorm2d module
        y2 = torch.nn.functional.batch_norm(y2, self.bn.running_mean, self.bn.running_var, self.bn.weight, self.bn.bias, False, 0.1, 0.000000100000000020000000000000000, False)
        output1 = self.conv2(y2)
        return output1

# Inputs to the model
x2 = 0.1*torch.randn(1, 1, 4, 4)
