
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t1 = torch.nn.ConvTranspose2d(3, 3, kernel_size=(3, 3), stride=(1, 2), bias=False)
        self.conv_t2 = torch.nn.ConvTranspose2d(3, 3, kernel_size=(3, 5), stride=(1, 1), bias=False)
    def forward(self, x1):
        v1 = self.conv_t1(x1)
        v2 = torch.sigmoid(v1)
        v3 = self.conv_t2(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 333, 444)
