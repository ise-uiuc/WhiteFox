
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        conv1 = torch.nn.ConvTranspose2d(3, 16, 3)
        bn1 = torch.nn.BatchNorm2d(16)
        conv2 = torch.nn.Conv2d(3, 16, 3)
        bn2 = torch.nn.BatchNorm2d(16)
        self.op = torch.nn.Sequential(
            conv1, bn1, nn.ReLU(), conv2, nn.Conv2d(16, 3, 3))
    def forward_(self, x):
        x = self.op(x)
        return x
# Inputs to the model
x = torch.randn(1, 3, 3, 3)
