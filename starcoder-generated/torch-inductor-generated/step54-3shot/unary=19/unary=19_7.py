
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(3, 16)
        self.fc2 = torch.nn.Linear(16, 1)
 
    def forward(self, inputs):
        x = self.fc1(inputs)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 3)
