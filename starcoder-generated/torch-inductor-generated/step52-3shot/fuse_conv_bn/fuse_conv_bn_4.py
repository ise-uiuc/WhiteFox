
# It does not generate the pattern as the input has a dimension mismatch
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        conv = torch.nn.Conv2d(1, 3, 3)
        self.conv = torch.nn.Sequential(conv, torch.nn.Conv2d(3, 3, 3))
        self.bn = torch.nn.BatchNorm2d(3)
        conv.register_backward_hook(self.backward_hook)
    def forward(self, x3):
        x = self.conv(x3)
        y = self.bn(x)
        z = self.conv(y)
        return z
    def backward_hook(self, module, grad_in, grad_out):
        # print(grad_out[0].size())
        print(grad_out[0].shape)
# Inputs to the model
x3 = torch.randn(2, 1, 4, 4)
