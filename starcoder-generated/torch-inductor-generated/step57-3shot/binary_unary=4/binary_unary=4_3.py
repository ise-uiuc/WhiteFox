
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(5, 10, bias=False)
        self.linear2 = torch.nn.Linear(10, 5, bias=False)
 
    def forward(self, x1, x2 = torch.randn(1, 5)):
        v1 = self.linear1(x1)
        v2 = v1 + x2
        v3 = F.relu(v2)
        v4 = self.linear2(v3)
 
        return v4

# Initializing the model
m2 = Model()

# Input to the model
x = torch.randn(1, 5)

# Call the model
