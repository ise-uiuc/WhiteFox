
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = torch.nn.Sequential(torch.nn.ConvTranspose2d(5, 10, 3, stride=1, padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1), padding=1))
        self.block2 = torch.nn.Conv2d(10, 8, 3, bias=False, padding=1)
    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = torch.sigmoid(x2)
        return x3
# Inputs to the model
x = torch.randn(1, 5, 32, 32)
