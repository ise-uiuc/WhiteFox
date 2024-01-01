
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(224*224*3, 1000)
        self.linear2 = torch.nn.Linear(224*224*3, 1000)
        self.linear3 = torch.nn.Linear(224*224*3, 1000)
 
    def forward(self, x1):
        v1 = self.linear1(x1.flatten())
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        v4 = self.linear2(x1.flatten())
        v5 = torch.sigmoid(v4)
        v6 = v4 * v5
        v7 = self.linear3(x1.flatten())
        v8 = torch.sigmoid(v7)
        v9 = v7 * v8
        return v3 + v6 + v9

# Initializing model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
