
class Model(torch.nn.Module):
    def __init__(self, dim, sequence_length, heads, dropout_p):
        super().__init__()
        self.dim = dim
        self.sequence_length = sequence_length
        self.heads = heads
        self.dropout_p = dropout_p
        self.scale_factor = math.sqrt(self.dim)
        self.to_query = torch.nn.Linear(self.dim, self.dim)
        self.to_key = torch.nn.Linear(self.dim, self.dim)
        self.to_value = torch.nn.Linear(self.dim, self.dim)
        self.dropout = torch.nn.Dropout(self.dropout_p)
 
    def forward(self, query, key, value):
        q = self.to_query(query)
        k = self.to_key(key)
        v = self.to_value(value)
        q *= self.scale_factor
        q = q.view(-1, self.heads, self.sequence_length, self.dim)
        k = k.view(-1, self.heads, self.sequence_length, self.dim)
        v = v.view(-1, self.heads, self.sequence_length, self.dim)
 
        scaled_qk = torch.matmul(q, k.transpose(-2, -1))
 
        softmax_qk = torch.nn.functional.softmax(scaled_qk, dim=-1)
        dropout_qk = self.dropout(softmax_qk)
        output = torch.matmul(dropout_qk, v)
 
        output = output.view(-1, self.heads * self.sequence_length, self.dim)
        output = self.dropout(output)
        return output

# Initializing the model
m = Model(dim=20, sequence_length=32, heads=12, dropout_p=0.05)

# Inputs to the model
query = torch.randn(1, 20, 32)
key = torch.randn(1, 20, 32)
value = torch.randn(1, 20, 32)
