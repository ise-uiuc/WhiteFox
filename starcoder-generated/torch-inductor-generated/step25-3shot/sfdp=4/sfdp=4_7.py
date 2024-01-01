
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, v1, v2):
        qk = v1 @ v2.transpose(-2, -1)
        return 0

# Initializing the model
m = Model()

# Inputs to the model
v1 = torch.randn(1, 8, 128, 1028)
v2 = torch.randn(1, 8, 128, 1028)
