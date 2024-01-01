
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(10, 10), stride=(1, 1))
        self.conv2 = torch.nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
        self.conv3 = torch.nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv4 = torch.nn.Conv2d(64, 8, kernel_size=(1, 1), stride=(1, 1))
 
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        v = out * 0.441
        t = torch.tanh(out)
        v = v + t
        v = v / 3.734
        return v
    