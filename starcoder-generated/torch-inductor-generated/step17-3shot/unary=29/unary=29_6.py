
class Model(torch.nn.Module):
    def __init__(self, min_value=3.2, max_value=3.2):
        super(Model, self).__init__()
        self.softsign = torch.nn.Softsign()
        self.leaky_relu = torch.nn.LeakyReLU()
        self.max_pool2d = torch.nn.MaxPool2d(2, stride=2, padding=0)
        self.conv2d = torch.nn.Conv2d(8, 18, 3, stride=1, padding=1)
        self.act_4 = torch.nn.Tanh()
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x3):
        v1 = self.conv2d(x3)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        v4 = self.softsign(v3)
        v9 = self.leaky_relu(v4)
        v11 = self.max_pool2d(v9)
        v13 = self.act_4(v11)
        return v13
# Inputs to the model 
x3 = torch.randn(1, 8, 224, 224)
