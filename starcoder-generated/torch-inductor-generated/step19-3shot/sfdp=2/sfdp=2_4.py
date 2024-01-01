
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = torch.nn.Linear(4, 8)
        self.dropout = torch.nn.Dropout(0.25)
 
    def forward(self, queries, keys, values):
        qkv = self.qkv(queries).chunk(3, dim=-1) # Split the queries into query, key, and values
        q, k, v = [e.squeeze(dim=0) for e in qkv] # Get the last dimension of the output of the first linear layer. Then remove the redundant batch size dimension inserted by chunk
        qkv = torch.matmul(q, k.transpose(-2, -1)) # Compute the dot product of the queries and the keys
        scaled_qkv = qkv.div(100)
        softmax_qkv = scaled_qkv.softmax(dim=-1) # Apply softmax to the scaled dot product
        dout_qkv = self.dropout(softmax_qkv) # Apply dropout to the softmax output
        output = torch.matmul(dout_qkv, v) # Compute the dot product of the dropout output and the values
        return output

# Initializing the model
m = Model()

# Inputs to the model
queries = torch.randn(1, 5, 4)
keys = torch.randn(4, 6, 4)
values = torch.randn(4, 6, 8)
