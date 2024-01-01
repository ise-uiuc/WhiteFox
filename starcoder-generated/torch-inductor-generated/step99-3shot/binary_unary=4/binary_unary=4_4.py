
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(8, 4)
 
    def forward(self, x1, other=None):
        v1 = self.fc(x1)
        v2 = v1 + other if not other is None else v1
        v3 = torch.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8)
# Specify the value of the `other` tensor
x2 = torch.randn(1, 4)
