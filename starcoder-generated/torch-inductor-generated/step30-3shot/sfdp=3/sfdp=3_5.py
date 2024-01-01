
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x):
        q = x * x # Generate the query tensor
        k = x * x # Generate the key tensor
        s = torch.matmul(q, k.transpose(-2, -1)) # Compute the dot product of the query and key tensors
        ssf = s * 0.5 # Scale the dot product by a factor
        smf = torch.softmax(ssf, dim=-1) # Apply softmax to the scaled dot product
        dm = torch.nn.functional.dropout(smf, p=0.1) # Apply dropout to the softmax output
        v = dm * s # Scale the dot product by a factor
        return v

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 32, 100)
