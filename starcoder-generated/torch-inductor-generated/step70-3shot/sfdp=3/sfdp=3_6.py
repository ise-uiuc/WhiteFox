
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2):
        v1 = torch.matmul(x1, x2.transpose(-2, -1)) # Apply a linear layer and pass the input tensors as the arguments
        v2 = v1 * 0.7071067811865476
        v3 = v2.softmax(dim=-1) # Apply softmax
        v4 = torch.nn.functional.dropout(v3, p=0.5)
        v5 = torch.matmul(v4, x2)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(16, 32, 8)
x2 = torch.randn(16, 8, 256)
