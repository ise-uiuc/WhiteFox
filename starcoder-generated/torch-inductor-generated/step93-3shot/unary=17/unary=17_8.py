
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Conv2d(16, 10, (1, 1), stride=(1, 1))
        self.conv2 = torch.nn.Conv2d(10, 16, kernel_size=(1, 1), stride=(1, 1))
    def forward(self, x):
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        x = self.conv2(x)
        return x
# Inputs to the model
x = torch.randn(2, 16, 16, 16)
