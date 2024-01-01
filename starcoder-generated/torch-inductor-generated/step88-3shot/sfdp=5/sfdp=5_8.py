
class Model(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 7, 3)
        self.conv2 = torch.nn.Conv1d(7, 10, 5, 2)
    def forward(self, x1, x2):
        return self.conv2(ReLU()(self.conv1(x1)))
# Inputs to the model
x1 = torch.randn(1, 10, 30)
x2 = torch.randn(1, 4, 20)
