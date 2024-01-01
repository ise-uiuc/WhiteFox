
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model1 = ConvBlock(4, 60, 3, 2,'relu')
        self.model2 = torch.nn.ConvTranspose2d(38, 20, kernel_size=(2, 2), stride=(2, 2))
    def forward(self, x26):
        r2 = self.model1(x26)
        r3 = self.model2(r2)
        return r3
# Inputs to the model
x26 = torch.randn(2, 4, 16, 18)
