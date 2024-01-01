
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = torch.nn.Conv2d(79, 1, 4, stride=1, padding=0)
        self.linear1 = torch.nn.Linear(513, 305)
        self.linear2 = torch.nn.Linear(305, 232)
    def forward(self, x1):
        v1 = self.conv0(x1)
        v2 = v1 + torch.tensor(11, requires_grad=True).cuda()
        v3 = v2.mean((-2, -1), keepdim=True)
        v4 = torch.clamp(v3, 0, 9)
        v5 = self.linear1(x1.flatten(1))
        v6 = self.linear2(v5)
        return torch.sum(v6 * v4)
# Inputs to the model
x1 = torch.randn(10, 79, 32, 32)
