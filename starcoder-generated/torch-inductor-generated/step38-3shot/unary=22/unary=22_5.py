
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 64)
 
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1)
 
        self.relu = torch.nn.ReLU()
 
        self.linear2 = torch.nn.Linear(8, 8)
        
    def forward(self, x):
        v1 = self.linear(x)

        v2 = self.conv(x)
 
        v3 = self.relu(v2)
 
        v4 = self.linear2(v3)
 
        return v4

# Initializing the model
m = Model()

# Input to the model
x = torch.rand(1, 3)
