
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.c = nn.ConvTranspose2d(3 * 6, 16, 4, 2, 1) 
        self.t = nn.Tanh()
        self.g = nn.BatchNorm2d(16)
        self.l = nn.LeakyReLU()
    def forward(self, x):
        out = self.c(x)
        out = self.t(out)
        out = self.g(out)
        out = self.l(out)
        return out, out
# Input to the model
x = torch.randn(5, 3 * 6, 96, 96)
