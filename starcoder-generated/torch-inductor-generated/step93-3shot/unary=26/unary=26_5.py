
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        model = Sequential(OrderedDict([
            ('0', Conv2d(1, 4, kernel_size=3, stride=1, padding=1)),
            ('1', MaxPool2d(2, 2)),
            ('2', Conv2d(4, 8, kernel_size=3, stride=1, padding=1)),
        ]))
        self.model = model
    def forward(self, a):
        return self.model(a)
# Inputs to the model
a = torch.randn(1, 1, 50, 50)
