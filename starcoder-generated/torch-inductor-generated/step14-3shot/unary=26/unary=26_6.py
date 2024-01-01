
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = torch.nn.Conv2d(3, 6, 5)
        self.c2 = torch.nn.ConvTranspose2d(6, 16, 3)
        self.p1 = torch.nn.MaxPool2d(2, stride=2)
        self.relu1 = torch.nn.ReLU()
        self.relu2 = torch.nn.ReLU()
    def forward(self, x):
        x = self.c1(x)
        x_clone = self.c2(x)
        x_pool = self.p1(x)
        x_out1 = self.relu1(x)
        x_out2 = self.relu2(x_clone)
        x_out3 = self.relu2(x_pool)
        return x_out1 * x_out2 * x_out3
# Inputs to the model
x = torch.randn(1, 3, 20, 20)
