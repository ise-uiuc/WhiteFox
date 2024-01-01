
class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.modules.Conv2d(3, 2, 1, stride=1, padding=1, bias=False)
        )
    def forward(self, x):
        y = self.layer1(x)
        p = torch.nn.functional.max_pool2d(y, (3, 3), stride=(3, 3), padding=(3, 3))
        z = torch.max(y, p)
        return z
# End of model
# End of inputs

# Model begins
class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.modules.Conv2d(16, 10, 5, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(10, 20, 5, stride=1, padding=1),
            torch.nn.ReLU(inplace=True)
        )
        #self.layer2 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1)

    def forward(self, x):
        x = self.layer1(x)
        y = torch.max(x, 2)
        return y[0]
# End of model
# End of inputs