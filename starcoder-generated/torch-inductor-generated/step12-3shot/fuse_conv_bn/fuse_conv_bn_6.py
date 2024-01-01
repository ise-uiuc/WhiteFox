
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = torch.nn.Conv2d(4, 8, 3)
        self.conv1 = torch.nn.Conv2d(8, 8, 1)
        self.conv1[-1].bias = torch.nn.Parameter(torch.ones(8))
    def forward(self, input):
        x = self.conv0(input)
        return self.conv1(x) + torch.tensor([1., 2., 3., 4., 5., 5., 6., 7.])[None, :, None, None]
# Inputs to the model
x3 = torch.randn(2, 4, 8, 8)
