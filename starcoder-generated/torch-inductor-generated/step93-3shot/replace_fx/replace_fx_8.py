
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(3,3)
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        
        return torch.sigmoid(x)
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(2, 2)
    def forward(self, x):
        x = self.fc(x)
        x1 = torch.sigmoid(x)
        x2 = torch.sigmoid(x)
        return x1 + x2
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(2, 2)
    def forward(self, x):
        x = self.fc(x)
        x1 = torch.sigmoid(x)
        x2 = torch.sigmoid(x)
        return x1 + x2

# Inputs to the model
x = torch.randn(3)
