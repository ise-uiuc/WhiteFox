
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.model1 = torch.nn.Sequential()
        self.model1.add_module("conv1", torch.nn.Conv2d(1, 20, 5, 1))
        self.model1.add_module("pool1", torch.nn.MaxPool2d(2, 2))
        self.model1.add_module("conv2", torch.nn.Conv2d(20, 50, 5, 1))
        self.model1.add_module("pool2", torch.nn.MaxPool2d(2, 2))
        self.model1.add_module("fc1", torch.nn.Linear(4*4*50, 500))
        self.model1.add_module("relu1", torch.nn.ReLU())
        self.model1.add_module("fc2", torch.nn.Linear(500, 84))
        self.tanh = torch.nn.Tanh()
    def forward(self, x):
        y = self.model1(x)
        y = self.tanh(y)
        return y
# Inputs to the model
x = torch.randn(1, 1, 28, 28)
