
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.key = torch.nn.Parameter(torch.randn(128, 8, 2))
        self.query = torch.nn.Parameter(torch.randn(128, 8, 2, 2))
        self.value = torch.nn.Parameter(torch.randn(128, 8, 2, 2))
        self.dropout_p = 0.4
 
    def forward(self, x1):
        k = self.key
        q = self.query
        v = self.value
        qk = torch.matmul(q, k.transpose(-2, -1)) # Compute the dot product of the query and the key
        inv_scale_factor = math.sqrt(q.size(-1))
        scaled_qk = qk.div(inv_scale_factor) # Scale the dot product by the inverse scale factor
        softmax_qk = scaled_qk.softmax(dim=-1) # Apply softmax to the scaled dot product
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p) # Apply dropout to the softmax output
        output = dropout_qk.matmul(v) # Compute the dot product of the dropout output and the value
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
