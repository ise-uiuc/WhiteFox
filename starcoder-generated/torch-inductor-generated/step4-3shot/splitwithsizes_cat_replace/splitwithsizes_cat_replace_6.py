
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.split1 = torch.nn.Linear(4, 4)
 
    def forward(self, x1):
        v1, v2, v3, v4 = torch.split(x1, (self.n1, 3, 3, 3), dim=1)
        v1 = self.split1(v1)
        return (v1 + v2, v3, v4)

# Initializing the model
model = Model()

# Inputs to the model
x1 = torch.randn(3, 10)
res1, res2, res3 = model(x1)

