
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(...)
 
    def forward(self, x1, x2):
        v1 = self.fc1(x1)
        v2 = torch.matmul(x2, v1.transpose(0, 1))
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 5)
x2 = torch.randn(3, 5)
