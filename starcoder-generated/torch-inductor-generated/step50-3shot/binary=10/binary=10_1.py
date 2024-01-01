
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 3)
        self.linear2 = torch.nn.Linear(3, 1)
 
    def forward(self, x):
        v1 = self.linear1(x)
        v2 = torch.empty(len(x))
        for i in range(len(x)):
            v2[i] = v1[i][0] + self.linear2.weight[0][0]
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x = torch.tensor([[1, 2], [1, 2]])
