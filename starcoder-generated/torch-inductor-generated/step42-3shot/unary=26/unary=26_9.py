
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.t1 = torch.nn.ConvTranspose2d(95, 173, 2, stride=1, padding=1, bias=False)
        self.acti1 = torch.nn.ReLU(inplace=False)
        self.t2 = torch.nn.ConvTranspose2d(173, 93, 2, stride=1, padding=1, bias=False)
        self.acti2 = torch.nn.Identity()
        self.t3 = torch.nn.ConvTranspose2d(93, 1, 3, stride=1, padding=1, bias=False)
        self.acti3 = torch.nn.Sigmoid()
    def forward(self, x3):
        x1 = self.t1(x3)
        x4 = self.acti1(x1)
        x5 = self.t2(x4)
        x7 = self.acti2(x5)
        x6 = self.t3(x7)
        x8 = self.acti3(x6)
        return x8
# Inputs to the model
x3 = torch.randn(16, 95, 8, 9)
