
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(3, 9)
        self.linear2 = torch.nn.Linear(9, 4)
 
    def forward(self, x1):
        v1 = self.linear1(x1)
        v2 = v1 - torch.tensor([1.2, 2.3, 3.4, 4.5], dtype=torch.float32)
        return self.linear2(torch.relu(v2))

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
