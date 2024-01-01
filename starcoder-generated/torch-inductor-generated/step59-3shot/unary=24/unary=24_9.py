
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.t1 = torch.nn.Conv2d(in_channels=4, out_channels=5, kernel_size=(38, 30), stride=(1, 1), bias=True)
        self.t2 = torch.nn.Conv2d(in_channels=5, out_channels=5, kernel_size=(29, 27), stride=(1, 1), bias=True)
        self.t3 = torch.nn.Conv2d(in_channels=5, out_channels=5, kernel_size=(26, 21), stride=(1, 1), bias=True)
        self.b1 = nn.ReLU(inplace=True)
    def forward(self, x):
        x1 = self.t1(x)
        x1 = self.b1(x1)
        x2 = self.t2(x1)
        x2 = self.b1(x2)
        x3 = self.t3(x2)
        x3 = self.b1(x3)
        return x3
# Inputs to the model
x = torch.randn(155, 4, 111, 156)
