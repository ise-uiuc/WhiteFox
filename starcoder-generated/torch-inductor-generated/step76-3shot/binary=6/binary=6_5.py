
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(3, 3)
 
    def forward(self, x1):
        x2 = F.relu(self.linear1(x1))
        x3 = x2 - 1
        x4 = F.sigmoid(x3)
        return x4


# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64)
