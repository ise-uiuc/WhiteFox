
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose1 = torch.nn.ConvTranspose2d(1, 3, 1, stride=1, padding=0)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(3, 2, 5, stride=2)
        self.relu1 = torch.nn.ReLU6()
        self.relu2 = torch.nn.ReLU6()
        self.max_pool = torch.nn.MaxPool2d(1, 1)
    def forward(self, x1):
        t1 = self.relu1(self.conv_transpose1(x1))
        t2 = self.conv_transpose2(t1)
        t3 = torch.add(t2, 1)
        t4 = self.relu2(t3)
        t5 = self.max_pool(t4)
        return t5
# Inputs to the model
x1 = torch.randn(5, 1, 3, 1)
