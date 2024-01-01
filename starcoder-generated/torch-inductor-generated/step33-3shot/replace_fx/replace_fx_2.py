
class Module0(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0_weight = torch.nn.Parameter(torch.randn([1, 3, 3, 3]))
    def forward(self, x1):
        x2 = F.conv2d(x1, self.conv0_weight)
        return x2
# Inputs to the model
x1 = torch.randn(1, 1, 4, 4)
