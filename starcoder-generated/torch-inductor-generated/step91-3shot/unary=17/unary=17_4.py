
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convtranspose2d_1 = torch.nn.ConvTranspose2d(4, 4, (2, 2), stride=(2, 2), bias=False)
    def forward(self, input):
        v1 = self.convtranspose2d_1(input)
        return v1
# Inputs to the model
input = torch.randn(1, 4, 5, 5)
