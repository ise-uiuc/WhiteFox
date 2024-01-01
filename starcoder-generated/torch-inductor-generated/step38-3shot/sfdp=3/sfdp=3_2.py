
class Mo(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, t0):
        v0 = torch.matmul(t0, t0.transpose(-2, -1))
        v1 = v0.mul(-10000.0)
        v2 = torch.nn.functional.softmax(v1)
        v3 = torch.nn.functional.dropout(v2, 0.0)
        v4 = torch.matmul(v3, t0)
        return v4

# Initializing the model
m = Mo()

# Inputs to the model
t0 = torch.randn(8, 8, 20)
