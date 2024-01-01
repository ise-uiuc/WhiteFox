
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1= torch.nn.Conv2d(512, 10, (1, 1), stride=(1, 7), padding=(0, 0))
        self.conv2= torch.nn.Conv2d(512, 64, (1, 1), stride=(5, 1), padding=(2, 0))
    def forward(self, x):
        t1 = self.conv1(x)
        t2 = self.conv2(x)
        t3 = t1 - t2
        return t3
# Inputs to the model
x = torch.randn(2, 512, 50, 70)
