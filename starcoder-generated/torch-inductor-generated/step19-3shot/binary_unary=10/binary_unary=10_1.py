
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_1 = torch.nn.Linear(653, 653)
 
        self.relu_1 = torch.nn.ReLU()
 
    def forward(self, x):
        v1 = self.linear_1(x)
        y = self.relu_1(v1)
        return y

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 653)

