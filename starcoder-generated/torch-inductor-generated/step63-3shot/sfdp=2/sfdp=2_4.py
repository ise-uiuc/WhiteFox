
class ScaledDotProductAttention(torch.nn.Module):
    def __init__(self, dropout_p):
        super().__init__()
        self.dropout_p = dropout_p
 
    def forward(self, query, key, value, inv_scale_factor):
        qk = torch.matmul(query, key.transpose(-2, -1)) # Compute the dot product of the query and the key
        scaled_qk = qk.div(inv_scale_factor) # Scale the dot product by the inverse scale factor
        softmax_qk = scaled_qk.softmax(dim=-1) # Apply softmax to the scaled dot product
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p) # Apply dropout to the softmax output
        output = dropout_qk.matmul(value) # Compute the dot product of the dropout output and the value
        return output
 
class Model(torch.nn.Module):
    def __init__(self, dropout_p=0.1):
        super().__init__()
        self.dropout_p = dropout_p
        self.attention1 = ScaledDotProductAttention(self.dropout_p)
 
    def forward(self, x):
        v1 = self.attention1(x, x, x, 1.0 / math.sqrt(x.shape[-1]))
        v2 = self.attention1(v1, v1, v1, math.sqrt(8.0))
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 512, 64)
x2 = torch.randn(1, 3, 64, 128)
x3 = torch.randn(1, 3, 128, 256)
