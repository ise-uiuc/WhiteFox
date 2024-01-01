
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(3, 5), stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(64, 128, kernel_size=(3, 5), stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        return v2
# Inputs to the model
x1 = torch.randn(2, 3, 60, 90) # Change the size of input tensors if necessary, e.g. (3, 66, 66), (32, 28, 28), (1, 64, 64), etc.
