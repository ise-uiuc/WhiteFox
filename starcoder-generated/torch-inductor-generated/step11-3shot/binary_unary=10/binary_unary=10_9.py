
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 32)
 
    def forward(self, x2):
        v7 = self.linear(x2)
        v8 = v7 + x2.mean(dim=[-1], keepdims=True)
        v9 = F.relu(v8)
        return v9

# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(1, 8)
