
class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layers = nn.Conv2d(1, 3, kernel_size=(1,1))
    def forward(self, x):
        x = self.layers(x)
        x = x.mean(1)
        return x
# Inputs to the model
x = torch.randn(2, 1, 1, 1)
