
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(6, 6)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        vv2 = torch.rand_like(v1) # Use another tensor to be added to the output linear tensor
        v3 = v1 + xv2
        v4 = F.relu(v3)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 6)
