
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(64, 10)
        self.fc2 = torch.nn.Linear(10, 100)
        self.fc3 = torch.nn.Linear(100, 64)
 
    def forward(self, input, other):
        x = self.fc1(input)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = torch.add(x, other)
        x = self.fc3(x)
        x = torch.relu(x)
        return x
 
# Initializing model
m = Model()

# Input tensor x
input = torch.randn(128, 64)

# Keyword argument that is passed to the model's forward function
other = torch.randn(128, 64)

# Inputs to the model
