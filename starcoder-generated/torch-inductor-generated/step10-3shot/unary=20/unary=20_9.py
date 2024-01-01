
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.ConvTranspose2d(32, 64, kernel_size=(1, 1), stride=(1, 1))
    def forward(self, x):
        o1 = self.l1(x)
        o2 = nn.Sigmoid(o1)
        return o2
# Inputs to the model
x = torch.randn(1, 32, 32, 32)
