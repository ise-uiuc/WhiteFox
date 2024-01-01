
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convtranspose3 = torch.nn.ConvTranspose3d(1, 1, kernel_size=(2, 2, 1), stride=(2, 2, 1))
        self.relu4 = torch.nn.ReLU()
    def forward(self, x):
        v1 = self.convtranspose3(x)
        v2 = self.relu4(v1)
        return v2
# Inputs to the model
x = torch.randn(1, 1, 2, 2, 1)
