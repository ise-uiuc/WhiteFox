
class Model(torch.nn.Module):
    def __init__(self, num_heads, query_size, key_size, hidden_size, dropout_p):
        super().__init__()
        self.scale_factor = torch.sqrt(torch.tensor([key_size]).float())
        self.query = torch.nn.Linear(query_size, hidden_size)
        self.key = torch.nn.Linear(key_size, hidden_size)
        self.value = torch.nn.Linear(key_size, hidden_size)
        self.dropout = torch.nn.Dropout(dropout_p)
 
    def forward(self, query, key, value, dropout_p=0.0):
        q = self.query(query)
        k = self.key(key)
        v = self.value(value)
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(self.scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_output = self.dropout(softmax_qk)
        output = dropout_output.matmul(v)
        return output, softmax_qk

# Initializing the model
model = Model(12, 32, 32, 64, 0.0)

# Input tensors to the model.
query = torch.randn(2, 12, 32)
key = torch.randn(2, 12, 32)
value = torch.randn(2, 12, 32)

# Dropout probability for the dropout layer in the model.
dropout_p = 0.0
output, softmax_qk = model(query, key, value, dropout_p)
output2, softmax_qk2 = model(query, key, value, dropout_p)
