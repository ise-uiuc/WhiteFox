
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(256, 256, kernel_size=(1, 9), stride=(1, 1), bias=False)
        self.convtranspose1d2 = torch.nn.ConvTranspose1d(256, 256, kernel_size=(1,5), stride=(1,5))
    def forward(self, x1):
        x2 = torch.sigmoid(x1)
        v1 = self.conv1(x2)
        v2 = self.convtranspose1d2(v1)
        v3 = v1 * v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 256, 2800)
