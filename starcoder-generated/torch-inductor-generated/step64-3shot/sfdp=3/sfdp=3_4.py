
class Model(torch.nn.Module):
    def __init__(self, batch, dim, num_heads=8, dropout_p=0.5):
        super().__init__()
        self.batch, self.dim = batch, dim
        self.num_heads, self.dropout_p = num_heads, dropout_p
        self.scale_factor = self.dim ** -0.5

        self.to_qkv = torch.nn.Linear(self.dim, self.dim * 3, bias=False)
        self.to_out = torch.nn.Linear(self.dim, self.dim)

        self.dropout = torch.nn.Dropout(p=dropout_p) # Add dropout to the softmax output
 
    def forward(self, input):
        x = self.to_qkv(input).reshape(self.batch, -1, 3, self.num_heads, self.dim // self.num_heads).permute(2, 0, 3, 1, 4)
        query, key, value = x[0], x[1], x[2]

        qk = torch.matmul(query, key.transpose(-2, -1)) # Compute the dot product of the query and key tensors
        scaled_qk = qk.mul(scale_factor) # Scale the dot product by a factor
        softmax_qk = scaled_qk.softmax(dim=-1) # Apply softmax to the scaled dot product
        dropout_qk = self.dropout(softmax_qk) # Apply dropout to the softmax output
        out = dropout_qk.matmul(value) # Compute the dot product of the dropout output and the value tensor
        out = self.to_out(out.reshape(batch, -1, dim))

        return out

# Initializing the model
batch, dim, num_heads, dropout_p = 16, 64, 8, 0.2
x = torch.randn(batch, dim)
