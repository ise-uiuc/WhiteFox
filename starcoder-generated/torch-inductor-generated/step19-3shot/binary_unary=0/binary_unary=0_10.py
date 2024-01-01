
class LeakyReLUModel(torch.nn.Module):
    def __init__(self, channel_size, nonnegative=True, input_scale=1e-3):
        super().__init__()
        self.channel_size = channel_size
        self.nonnegative = nonnegative
        self.input_scale = input_scale
        self.conv1 = torch.nn.Conv2d(channel_size, channel_size, 3, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(channel_size)
    def forward(self, x):
        input_prescale = x * self.input_scale
        conv1 = self.conv1(input_prescale)
        bn = self.bn1(conv1)

        # There is not an appropriate API to construct a LeakyReLU layer. To simulate this layer, a small piece of Python code is added.
        relu = (self.nonnegative & (bn >= 0)) * (bn - 0.01 * bn * (bn < 0)) + self.nonnegative * bn

        relu *= (1.0 / self.input_scale)
        return relu

# Inputs to the model
input_size = 224
x = torch.randn((1, 128, input_size, input_size), requires_grad=True)

# Model begins
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = torch.nn.Conv2d(128, 512, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(512, 512, 1, stride=1, padding=0)
        self.fc = torch.nn.Linear(512, 512)
    
    def forward(self, x):
        feature1 = self.conv1(x).mean((-2, -1))
        feature2 = self.conv2(feature1)
        feature3 = feature2.mean((-2, -1))
        out = self.fc(feature3)
        return out
# Inputs to the model
x = torch.randn(2, 5, 10)
