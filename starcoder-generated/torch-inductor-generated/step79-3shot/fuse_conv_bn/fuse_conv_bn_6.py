
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(7)
        self.layer1 = torch.nn.Sequential(torch.nn.Conv2d(3, 6, 1), torch.nn.BatchNorm2d(6))
    def forward(self, x1):
        s1 = self.layer1(x1)
        return s1
# Inputs to the model
x1 = torch.randn(1, 3, 3, 3)
