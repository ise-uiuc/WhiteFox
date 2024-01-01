
class Model(torch.nn.Module):
    def __init__(self, dim=500):
        super().__init__()
        self.dim = dim
        self.fc1 = torch.nn.Linear(self.dim, self.dim)
        self.fc2 = torch.nn.Linear(self.dim, self.dim)
        self.fc3 = torch.nn.Linear(self.dim, self.dim)
        self.fc4 = torch.nn.Linear(self.dim, self.dim)
 
    def forward(self, x1, x2):
        y1 = self.fc1(x1) # Apply the first fully connected layer
        y2 = self.fc2(y1) # Apply the second fully connected layer
        y3 = self.fc3(y1 + x2) # Apply the third fully connected layer, the output of the second fully connected layer is added to the input tensor
        y4 = self.fc4(y3 + x2) # Apply the fourth fully connected layer, the output of the third fully connected layer is added to the input tensor
        return y4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 500)
x2 = torch.randn(1, 500)
