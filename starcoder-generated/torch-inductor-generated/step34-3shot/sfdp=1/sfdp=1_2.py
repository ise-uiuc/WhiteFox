
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.dropout_p=0.1
        self.scaling_factor = 10
    
    def forward(self, __input__, __input__1):
        
        v0 = torch.matmul(__input__, __input__1.transpose(-2, -1))
        v1 = v0.div(self.scaling_factor)
        v2 = torch.nn.functional.softmax(v1, dim = -1)
        v3 = torch.nn.functional.dropout(v2, p = self.dropout_p)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 32, 8)
key = torch.randn(1, 32, 8)
value = torch.randn(1, 32, 8)
