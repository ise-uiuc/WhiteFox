
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(9)
        self.layer = torch.nn.Sequential(torch.nn.Conv2d(1, 2, 1, bias=True), torch.nn.BatchNorm2d(2))
    def forward(self, x3):
        s3 = self.layer(x3)
        return s3
# Inputs to the model
x3 = torch.randn(1, 1, 2, 2)
