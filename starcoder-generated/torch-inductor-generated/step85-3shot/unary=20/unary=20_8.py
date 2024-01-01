
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convT1 = torch.nn.ConvTranspose2d(42, 7, kernel_size=(10, 10), stride=(1, 1), padding=(5, 5), dilation=(2, 2))
        self.convT2 = torch.nn.ConvTranspose2d(7, 7, kernel_size=(1, 10), stride=(10, 1), padding=(1, 5), dilation=(10, 2))
    def forward(self, x):
        v1 = self.convT1(x)
        v2 = torch.sigmoid(v1)
        v3 = self.convT2(v2)
        v4 = torch.sigmoid(v3)
        return v4
# Inputs to the model
x = torch.randn(1, 42, 111, 111)
