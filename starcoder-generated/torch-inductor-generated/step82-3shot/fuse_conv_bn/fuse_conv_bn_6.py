
class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 192, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    def forward(self, x):
        x = torch.nn.functional.relu(self.conv(x), inplace=False)
        return x
# Inputs to the model
x = torch.randn(1, 3, 224, 224)
