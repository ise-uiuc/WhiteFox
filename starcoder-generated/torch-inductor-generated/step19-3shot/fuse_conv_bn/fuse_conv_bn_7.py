
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(7)
        self.layer = torch.nn.Sequential(torch.nn.Conv2d(4, 3, 3, bias=True), torch.nn.BatchNorm2d(3), torch.nn.ReLU6())
    def forward(self, x1):
        s1 = self.layer(x1)
        return s1 + s1
# Inputs to the model
x1 = torch.randn(1, 4, 4, 4)
