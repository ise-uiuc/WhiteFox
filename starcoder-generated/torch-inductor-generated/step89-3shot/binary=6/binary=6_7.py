
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        size = 64
        self.fc0 = torch.nn.Dropout(0.2)
        self.fc1 = torch.nn.Linear(size, size // 2)
        self.fc2 = torch.nn.Linear(size // 2, size * 4)
        self.fc3 = torch.nn.Linear(size * 4, size * 4)
        self.fc4 = torch.nn.Linear(size * 4, num_classes)
 
    def forward(self, x):
        x = x1 = self.fc0(x)
        v1 = x1 = self.fc1(x)
        v2 = x1 = self.fc2(x)
        v3 = x1 = self.fc3(x)
        x1 = self.fc4(x3)
        return x3

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(17, 32, 28, 28)
