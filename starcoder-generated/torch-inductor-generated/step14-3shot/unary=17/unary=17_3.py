
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.ConvTranspose2d(2, 2, 3, bias=True)
        self.conv2d_1 = torch.nn.ConvTranspose2d(5, 2, kernel_size=(5, 3), bias=False)
        self.conv2d_2 = torch.nn.ConvTranspose2d(4, 5, kernel_size=(2, 4), stride=(1, 2), bias=None)
    def forward(self, x1):
        v1 = self.conv2d(x1)
        v2 = v1.detach()
        v3 = self.conv2d_1(v2)
        v4 = self.conv2d_2(v1)
        v5 = v4.detach()
        return v5
# Inputs to the model
x1 = torch.randn(7, 4, 20, 10)
