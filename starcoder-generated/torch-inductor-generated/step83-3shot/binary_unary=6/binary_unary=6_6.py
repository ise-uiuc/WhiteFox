
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)
 
    def forward(self, x1, i):
        v1 = self.linear(x1)
        v2 = v1 - i
        v3 = F.relu(v2)
        return v3

# Initializing the model
m = Model()
 
# Inputs to the model
x1 = torch.randn(1, 10)
i = torch.randint(-6, 6, [1])
