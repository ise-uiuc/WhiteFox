
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.convt1 = torch.nn.ConvTranspose2d(1, 32, 3, 2)
        self.convt2 = torch.nn.ConvTranspose2d(32, 64, 3, 1)
        self.convt3 = torch.nn.ConvTranspose2d(64, 3, 3, 2)
    def forward(self, x):
        t1 = torch.tanh(self.convt1(x))
        t2 = self.convt2(t1)
        t3 = self.convt3(t2)
        return torch.tanh(t3)
# Inputs to the model
x = torch.randn(1, 1, 32, 32)
