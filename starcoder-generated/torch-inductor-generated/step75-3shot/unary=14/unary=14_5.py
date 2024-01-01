
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convtranspose6 = torch.nn.ConvTranspose2d(3, 3, kernel_size=(4, 2), stride=(2, 1), padding=(0, 1))
    def forward(self, x):
        v1 = self.convtranspose6(x)
        v2 = torch.nn.Sigmoid()(v1)
        #v3 = v1 * v2
        return v2
# Inputs to the model
x1 = torch.rand(1, 3, 6, 3)
