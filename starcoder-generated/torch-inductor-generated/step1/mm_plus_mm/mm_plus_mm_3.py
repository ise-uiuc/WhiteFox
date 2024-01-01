
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.p1 = torch.nn.Parameter(1)
        self.p2 = torch.nn.Parameter(2)
        
    def forward(self, x, y):
        v1 = torch.mm(x, y) # Multiply to tensors
        v2 = torch.mm(x, y) # Multiply to tensors
        v3 = v1 + v2 # Add two tensors
        return v3
    
# Initializing the model
m = Model()

def __init__():
    x = torch.randn(8)
    y = torch.randn(8)
    return {
            "inputs": [x, y]
            }
