
class Model(torch.nn.Module):
    def __init__(self, hidden_size, head_num, dropout_p):
        super().__init__()
        self.query = torch.nn.Linear(hidden_size, hidden_size)
        self.key = torch.nn.Linear(hidden_size, hidden_size)
        self.value = torch.nn.Linear(hidden_size, hidden_size)
        self.dropout_p = dropout_p
 
    def forward(self, query, key, value, inv_scale_factor):
        qk = torch.matmul(self.query(query), self.key(key).transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, self.dropout_p)
        output = dropout_qk.matmul(self.value(value))
        return output, softmax_qk
 
# Initializing the model
m = Model(100, 4, 0.1)

# Inputs to the model
query = torch.randn(4, 50, 100)
key = torch.randn(4, 10, 100)
value = torch.randn(4, 10, 100)
inv_scale_factor = torch.full((4, 50, 10), 10, dtype=torch.float32)
__output__, __softmax_qk__ = m(query, key, value, inv_scale_factor)

