
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.relu = nn.ReLU()
        self.max1 = nn.MaxPool2d(kernel_size=2, stride=2)
        model = nn.Sequential(nn.Conv2d(1, 20, 5, 1), nn.BatchNorm2d(20), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2), nn.Conv2d(20, 20, 5, 1), nn.BatchNorm2d(20), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))
        model_temp = [model[4], model[7]]
        self.model = nn.Sequential(model_temp)
        self.conv = nn.Conv2d(20, 50, 3, 1)
        model_temp = [model[0], self.conv, model[3], self.max1, model[6], self.relu]
        self.model1 = nn.Sequential(*model_temp)
    def forward(self, x):
        y = self.model1(x)
        z = self.model(y)
        return z
# Inputs to the model
x = torch.randn(2, 1, 28, 28)
