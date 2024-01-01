
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q = torch.nn.Linear(1000, 1000)
        self.k = torch.nn.Linear(1000, 1000)
        self.v = torch.nn.Linear(1000, 1000)
 
    def forward(self, query, key, value, scale_factor, dropout_p):
        q = self.q(query) # Compute the query
        k = self.k(key) # Compute the key
        v = self.v(value) # Compute the value
        qk = torch.matmul(q, k.transpose(-2, -1)) # Compute the dot product of the query and key tensors
        scaled_qk = qk.mul(scale_factor) # Scale the dot product by a factor
        softmax_qk = scaled_qk.softmax(dim=-1) # Apply softmax to the scaled dot product
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p) # Apply dropout to the softmax output
        output = dropout_qk.matmul(v) # Compute the dot product of the dropout output and the value tensor
        return dropout_qk

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(5, 1000)
key = torch.randn(10, 1000)
value = torch.randn(10, 1000)
scale_factor = 1 / math.sqrt(1000)
dropout_p = 0.1
