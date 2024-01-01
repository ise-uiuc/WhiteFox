
class Model(torch.nn.Module):
    def __init__(self, num_heads): # Initialize num_heads
        super().__init__()
        self.proj0 = torch.nn.Linear(80, num_heads * 8) # Project to a query, key, and value of 8-dimensional vectors
        self.proj1 = torch.nn.Linear(num_heads * 8, num_heads * 8)
 
    def scaled_dot_product_attention(self, q, k, v): # Attention core with different query, key, and value
        inv_scale_factor = np.power(k.shape[-1], -0.5) # Compute the inverse factor for the dot product of query and key
        qk = torch.matmul(qk, k.transpose(-2, -1)) # Compute the dot product of query and key
        scaled_qk = qk.div(inv_scale_factor) # Scale the dot product by the inverse factor
        softmax_qk = F.softmax(scaled_qk, dim=-1) # Apply softmax to the scaled dot product
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p) # Applay dropout to the softmax output
        output = torch.matmul(dropout_qk, v) # Compute the dot product of the dropout output and the value
        return output
 
    def forward(self, x1):
        v1 = self.proj0(x1)
        qk = self.proj1(v1) # Query, key, and value
        v2 = self.scaled_dot_product_attention(qk, qk, qk)
        return v2

# Initializing the model
m = Model(8)

# Inputs to the model
x1 = torch.randn(2, 80)
