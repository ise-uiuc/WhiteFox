
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2 = torch.nn.Conv2d(3, 1, kernel_size=(1, 3), stride=(2, 1), padding=(0, 1))
    def forward(self, x1, other=1, padding1=2, padding2=2):
        x1 = self.conv2(x1)
        x2 = x1 + other
        x3 = x2 + padding1
        x4 = x3 + padding2
        return x4
# Input values to the model
x1 = torch.randn(1, 3, 64, 64)
