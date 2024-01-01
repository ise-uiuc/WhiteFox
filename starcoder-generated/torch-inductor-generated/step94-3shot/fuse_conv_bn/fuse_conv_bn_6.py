
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1)
        self.conv = torch.nn.Conv2d(27, 27, 1)
        torch.manual_seed(1)
        self.batch_norm = torch.nn.BatchNorm2d(27)
    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        return x
# Inputs to the model
x = torch.randn(1, 27, 32, 32)
