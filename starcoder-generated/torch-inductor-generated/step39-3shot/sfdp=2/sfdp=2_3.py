
class Model(torch.nn.Module):
    def __init__(self, hidden_size, num_heads, dropout_p=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.query = torch.nn.Linear(hidden_size, hidden_size)
        self.key = torch.nn.Linear(hidden_size, hidden_size)
        self.value = torch.nn.Linear(hidden_size, hidden_size)
        self.matmul1 = torch.matmul
        self.matmul2 = torch.matmul
     
        weight = torch.empty(num_heads, hidden_size // num_heads, hidden_size, dtype=torch.float32)
        nn.init.xavier_uniform_(weight)
        bias = torch.empty(num_heads, hidden_size // num_heads, dtype=torch.float32)
        nn.init.normal_(bias)
        self.matmul1.weight = nn.Parameter(weight)
        self.matmul1.bias = nn.Parameter(bias)
        self.matmul2.weight = nn.Parameter(torch.transpose(weight, 1, 2)) # Use the transpose weight to form the inverse
        self.matmul2.bias = nn.Parameter(0)
        self.softmax = torch.nn.Softmax(dim=3)
        self.dropout = torch.nn.Dropout(p=dropout_p)

    def forward(self, query_tensor, key_tensor, value_tensor, query_mask, input_mask):
        q = self.query(query_tensor).view(query_tensor.shape[0], query_tensor.shape[1], -1).transpose(1, 2) # Compute the query
        k = self.key(key_tensor).view(key_tensor.shape[0], key_tensor.shape[1], -1) # Compute the key
        v = self.value(key_tensor).view(key_tensor.shape[0], value_tensor.shape[1], -1) # Compute the value
        qk = self.matmul1(q, torch.transpose(k, 1, 2)).div(math.sqrt(q.shape[-1])) # Compute the dot product of the query and the key and then scale it by the square root of the size of the query
        scaled_qk = qk.masked_fill(mask=query_mask, value=float('-inf')).softmax(dim=-1) # Set the masked elements in the dot product to NEGATIVE INF and then apply softmax
        dropout_qk = self.dropout(scaled_qk) # Apply dropout to the softmax output
        output = self.matmul2(dropout_qk, v.transpose(1, 2)) # Compute the dot product of the dropout output and the value
        return output, query_mask, input_mask

# Initializing the model
hidden_size = 64
num_heads = 8
m = Model(hidden_size=hidden_size, num_heads=num_heads)

# Inputs to the model
query_tensor = torch.randn(1, hidden_size)
key_tensor = torch.randn(2, hidden_size)
value_tensor = torch.randn(2, hidden_size)
query_mask = torch.zeros((1, 8, 2))
input_mask = torch.zeros((2, 8, 2))

__output__, __query_mask__, __input_mask__ = m(query_tensor, key_tensor, value_tensor, query_mask, input_mask)

# Outputs
print("output:", __output__)
print("query_mask:", __query_mask__)
print("input_mask:", __input_mask__)

