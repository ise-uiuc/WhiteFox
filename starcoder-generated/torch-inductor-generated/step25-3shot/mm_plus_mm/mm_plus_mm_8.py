
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(10, 10)  
        self.fc2 = torch.nn.Linear(10, 10)
    def forward(self, x):
        x1 = self.fc1(x)
        x2 = self.fc2(x1)
        return x2
# Inputs to the model
x = torch.randn(1, 10)
