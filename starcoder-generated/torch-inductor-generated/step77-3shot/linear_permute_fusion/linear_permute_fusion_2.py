
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv3x3 = torch.nn.Conv2d(1, 2, 3).cuda()
    def forward(self, x1):
        v1 = torch.nn.functional.pad(x1, (1, 1, 1, 1))
        v1 = v1.permute(0, 2, 3, 1).cuda()
        v1 = self.conv3x3(v1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        return v2
# Inputs to the model
x1 = torch.randn(3, 1, 5, 5)
