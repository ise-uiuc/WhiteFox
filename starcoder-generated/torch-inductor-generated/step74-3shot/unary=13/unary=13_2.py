
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(8, 8)
        self.fc2 = torch.nn.Linear(8, 8)
 
     def forward(self, x):
         v1 = self.fc1(x)
         v2 = torch.sigmoid(v1)
         v3 = v1 * v2
         v4 = self.fc2(v3)
         v5 = v4 * torch.sigmoid(v4)
         v6 = self.fc2(v5)
         return v6

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 8)
