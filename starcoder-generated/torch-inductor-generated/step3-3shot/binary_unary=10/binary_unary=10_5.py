
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.weights = torch.arange(1,17,dtype=torch.float)
        self.linear = torch.nn.Linear(16, 1)
 
    def forward(self, x1, x2):
        o1 = self.linear(x1).reshape(x1.shape[0], 16)
        o2 = o1 + self.weights.reshape(1, 16)
        o3 = torch.relu(o2)
        return o3

# Initialize the model
m = Model()

# Inputs of the model
x1 = torch.randn(3, 16)
x2 = torch.randn(3, 16)

