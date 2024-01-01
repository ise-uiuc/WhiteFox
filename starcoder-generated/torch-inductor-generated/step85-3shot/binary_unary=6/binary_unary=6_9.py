
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(19, 8, bias=False)
 
        self.linear2 = torch.nn.Linear(3, 8, bias=False)
        self.linear3 = torch.nn.Linear(8, 3, bias=False)
 
    def forward(self, x1, other):
        # 'other' is generated during the training process when applying 'Model'.
        v1 = self.linear(x1)
        v2 = v1 - other
        v3 = torch.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 19)
