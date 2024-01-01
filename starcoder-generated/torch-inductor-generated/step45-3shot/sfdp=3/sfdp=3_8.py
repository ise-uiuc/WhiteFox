
class scaled_dot_product_attention(torch.nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        # Use nn.LayerNorm as a pre-processing step
        self.scale_factor = torch.nn.Parameter(0.1, requires_grad=True)
        self.dropout = torch.nn.Dropout(dropout)
 
    def forward(self, query, key, value):
        # Compute the dot product of the query and key tensors
        qk = torch.matmul(query, key.transpose(-2, -1))
        # Scale the dot product by a factor
        scaled_qk = qk.mul(self.scale_factor)
        # Apply softmax to the scaled dot product
        softmax_qk = scaled_qk.softmax(dim=-1)
        # Apply dropout to the softmax output
        dropout_qk = self.dropout(softmax_qk)
        # Compute the dot product of the dropout output and 
        # the value tensor
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = scaled_dot_product_attention(dropout=0.1)

# Inputs to the model
query = torch.randn(1, 24, 512)
key = torch.randn(1, 8, 512)
value = torch.randn(1, 8, 512)
