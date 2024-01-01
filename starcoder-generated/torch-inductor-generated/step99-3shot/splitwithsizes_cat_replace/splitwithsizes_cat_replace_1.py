
class Model(torch.nn.Module):
    def __init__(self):
        super(torch.nn.Module, self).__init__()
        self.feature1 = torch.nn.Conv2d(3, 1, 1, 1, 0)
    def forward(self, input):
        x81 = self.feature1(input)
        x87 = torch.split(x81, [1, 1, 1], dim=1)
        x324 = torch.cat(x87, dim=1)
        return (x324, x87)
# Inputs to the model
x1 = torch.randn(64, 3, 64, 64)
