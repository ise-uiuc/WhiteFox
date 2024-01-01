
class MyModule(torch.nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()

    def forward(self, x):
        if x.shape[1] == 0 or x.shape[1] == 3 or x.shape[1] == 5:
            return x
        if hasattr(torch.nn.functional,'max_pool1d'):
            return torch.nn.functional.max_pool1d(x, 2, 2)
        else:
            return torch.max_pool1d(x, 2, 2)

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 2)
        self.pool2d = MyModule()
        self.conv2 = torch.nn.Conv2d(3, 3, 2, padding=(3, 1), dilation=(3, 2))
    def forward(self, x):
        x = self.conv(x)
        x = self.pool2d(x)
        x = self.conv2(x)
        return x
# Inputs to the model
x = torch.randint(0, 100, (1, 3, 10, 10))
