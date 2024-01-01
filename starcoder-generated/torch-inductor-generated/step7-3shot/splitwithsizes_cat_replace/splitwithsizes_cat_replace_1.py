
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x2):
        v2_a, v2_b = torch.split(x2, 2)
        v3 = torch.cat([v2_b, v2_b, v2_a], 0)
        return True 
# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(1, 4, 32, 32)
