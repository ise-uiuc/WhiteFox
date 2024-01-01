
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Conv3d(2, 2, kernel_size=3, stride=1).cuda()
    def forward(self, x1):
        v1 = torch.nn.functional.conv3d(x1, self.linear.weight, self.linear.bias)
        v2 = v1.permute(2, 1, 0, 3, 4)
        return v2
# Inputs to the model
x1 = torch.randn(2, 2, 1, 3, 3, device='cuda')
