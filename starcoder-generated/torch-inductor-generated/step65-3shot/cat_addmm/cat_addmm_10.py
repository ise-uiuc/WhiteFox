
class Model_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(128, 97)
    def forward(self, x):
        x = self.layers(x)
        return x

class Model_2(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(97, 132)
    def forward(self, x):
        x = self.layers(x)
        return x

class Model_3(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers_1 = nn.Linear(54, 92)
        self.layers_2 = nn.Linear(92, 92)
    def forward(self, x):
        x = self.layers_1(x)
        x = self.layers_2(x)
        return x

class Model_4(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(92, 12)
    def forward(self, x):
        x = self.layers(x)
        return x
# Inputs to the model
x = torch.randn(2, 54)
