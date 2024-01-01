
class Model(torch.nn.Module):
    def __init__(self):
        print('Constructor')
        super().__init__()
        self.linear = torch.nn.Linear(3, 3)
 
    def forward(self, x1):
        v0 = None
        v1 = self.linear(x1)
        v2 = {} # Add other
        v3 = v1 + v2
        v4 = torch.relu(v3)
        return v4

# Initializing the model
m = Model()
# print(m)

print('Inputs to the model')
# Inputs to the model
x1 = torch.randn(1, 3)
x2 = torch.randn(1, 3)
