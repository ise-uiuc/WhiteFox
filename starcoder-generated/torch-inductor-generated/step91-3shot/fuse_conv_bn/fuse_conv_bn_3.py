
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block_sequence = torch.nn.Sequential(torch.nn.Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), torch.nn.BatchNorm2d(256), torch.nn.ReLU(), torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), torch.nn.BatchNorm2d(256), torch.nn.ReLU(), torch.nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False), torch.nn.BatchNorm2d(256))
        self.conv = torch.nn.Conv2d(256, 10, kernel_size=(1, 1), stride=(1, 1), bias=False)
    def forward(self, x, y):
        x = self.block_sequence(x)
        y = self.conv(y)
        x_1x1 = x + y
        output = x_1x1
        return output
# Inputs to the model
x = torch.randn(1, 512, 2, 2)
y = torch.randn(1, 10, 2, 2)
