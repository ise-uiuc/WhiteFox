
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = torch.nn.Conv2d(2, 2, 1).cuda()
        self.linear = torch.nn.Linear(6, 2)
    def forward(self, x1):
        v4 = x1
        v1 = self.conv0(v4)
        v2 = v1.permute(0, 3, 1, 2)
        v3 = v2.view(v2.size(0), -1)
        v5 = v3.cuda()
        v = torch.nn.functional.linear(v5, self.linear.weight, self.linear.bias)
        return v
# Inputs to the model
x1 = torch.randn(1, 2, 2, 2, device='cuda')
