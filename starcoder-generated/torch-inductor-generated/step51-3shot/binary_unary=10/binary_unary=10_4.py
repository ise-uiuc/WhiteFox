
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(1, 4, bias=True)
        self.linear2 = torch.nn.Linear(4, 1, bias=True)

    def forward(self, x1):
        # Define the operations
        v1 = self.linear1(x1)
        v2 = v1 + x1
        v3 = v2.relu()
        # Define the graph
        v4 = self.linear2(v3)
        return v4

# Initializing the model
m = Model()

# Input to the model
x1 = torch.randn(20, 1)
