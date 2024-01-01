
class Model(torch.nn.Module):
    def __init__(self):
            super().__init__()
            self._conv2d = torch.nn.Conv2d(1, 20, 5, 1)  # input has shape of (1, 1, 28, 28)
            self._max_pool2d = torch.nn.MaxPool2d(2, 2)
    def forward(self, x):
        x = self._conv2d(x)  # using self._conv2d is OK as it stores the parameters for conv
        x = self._max_pool2d(x)
        return x
# Inputs to the model
x = torch.randn(1, 1, 28, 28)
