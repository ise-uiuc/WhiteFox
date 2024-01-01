
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Fill in below
        self.linear1 = torch.nn.Linear(256, 256, False)
        self.linear2 = torch.nn.Linear(128, 128)
 
    def forward(self, x1, x2):
        # Fill in below
        v1 = self.linear1(x1)
        v2 = v1 + x2
        v3 = self.linear2(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 256)
x2 = torch.randn(1, 128)
