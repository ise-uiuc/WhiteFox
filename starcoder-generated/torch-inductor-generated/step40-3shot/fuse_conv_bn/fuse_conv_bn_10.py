
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        conv = torch.nn.Conv2d(3, 128, kernel_size=3)
        bn = torch.nn.BatchNorm2d(num_features=3)
        self.convbn = torch.nn.Sequential(conv, bn)
    def forward(self, x):
        output = self.convbn(x)
        return output
# Inputs to the model
x1 = torch.randn(1, 3, 10, 10)
