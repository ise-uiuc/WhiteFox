
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 4)
        self.linear2 = torch.nn.Linear(4, 5)
 
    def forward(self, x):
        v1 = self.linear1(x).mean(2, True).mean(3, True)
        v2 = self.linear2(v1)
        v3 = torch.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs
x = torch.Tensor(1, 2, 4, 4)
x.uniform_(-10, 10)
