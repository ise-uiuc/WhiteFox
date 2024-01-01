
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.linear = torch.nn.Linear(256, 1024)
 
    def forward(self, x):
        v1 = self.linear(x).flatten()
        v2 = v1 + v1
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
