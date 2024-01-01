
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(200, 2)
 
    def forward(self, x1, other1=None):
        v1 = self.linear(x1)
        if other1 is None:
            other1 = torch.nn.Parameter(torch.ones((200, )))
        v2 = v1 - other1
        v3 = torch.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 200)
