
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout_p = 0.5
 
    def forward(self, q1, k1, v1):
        scale_factor = (q1.size(-1) / k1.size(-1)) ** 0.25 # Determine the scaling factor
        qk = torch.matmul(q1, k1.transpose(-2, -1)) # Compute the dot product
        v2 = qk.mul(scale_factor) # Scale the dot product
        v3 = v2.softmax(dim=-1) # Apply softmax on the scaled dot product
        v4 = torch.nn.functional.dropout(v3, p=self.dropout_p) # Apply dropout
        v5 = torch.matmul(v4, v1) # Compute the dot product
        return v5

# Initializing the model
m = Model()

# Inputs to the model
q1 = torch.randn(1, 128, 1, 3212)
k1 = torch.randn(1, 128, 10, 3212)
v1 = torch.randn(1, 128, 10, 3212)
