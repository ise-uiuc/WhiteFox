
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(3, 5, kernel_size=(1,1), padding=(0, 0))
    def forward(self, x1):
        v1 = self.conv2d(x1)
        v2 = torch.sigmoid(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 10, 10)
