
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        v1 = []
        # Number of loops: 6
        for _ in range(6):
            # Number of loop variables = 1
            # Loop variable declarations: loopVar5
            v1.append(torch.mm(x, x))
        # End of generated for-loop
        return torch.cat(v1, 1)    
# Inputs to the model
x = torch.randn(6, 1)
