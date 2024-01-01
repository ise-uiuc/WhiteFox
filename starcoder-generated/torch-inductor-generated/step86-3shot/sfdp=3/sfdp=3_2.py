
class Attention(torch.nn.Module):
    def __init__(self, dim, heads=8, dropout_p=0.1) -> None:
        super().__init__()
        self.scale_factor = dim ** -0.5

        self.query_proj, self.key_proj, self.value_proj = nn.Linear(dim, dim), nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x1, x2 = 0):
        if not x2:
            x2 = torch.transpose(x1, 0, 1)
        q = self.query_proj(x1)
        k = self.key_proj(x2)
        v = self.value_proj(x2)
       
        qk = torch.matmul(q, k.transpose(-2, -1)) # (x1_size, dim, 1) x (x2_size, dim, dim) -> (x1_size, dim, dim)
        scaled_qk = qk.mul(self.scale_factor) # Scale the dot product by a factor
        softmax_qk = scaled_qk.softmax(dim=-1) # Apply softmax to the scaled dot product
        dropout_qk = self.dropout(softmax_qk) # Apply dropout to the softmax output
        output = dropout_qk.matmul(v) # Compute the dot product of the dropout output and the value tensor

        return output

# Initializing the model
m = Attention(256)

# Inputs to the model
x1 = torch.randn(256, 10)
x2 = 0 # or torch.randn(10, 256)
