
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(3, 10, (1, 4), stride=1, padding=(1, 1), bias=False)
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        y = self.conv_t(x)
        y1 = y > 0
        y2 = y * 2.513
        y3 = torch.where(y1, y, y2)
        y4 = self.relu(y3)
        return y4
# Inputs to the model
x = torch.randn(1, 3, 11, 13)
