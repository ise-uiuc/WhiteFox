
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(1, 64, 3, stride=2, padding=1)
        self.max_pooling = torch.nn.MaxPool2d(kernel_size=5, stride=2, padding=1)
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        a = self.conv_t(x)
        b = self.max_pooling(a)
        c = a > 0
        d = a * -10
        e = torch.where(c, a, d)
        f = self.relu(e)
        return f
# Inputs to the model
x = torch.randn(1, 1, 240, 240)
