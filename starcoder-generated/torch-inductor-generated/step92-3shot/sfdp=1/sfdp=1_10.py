
class Model(torch.nn.Module):
    def __init__(self, hidden_size, num_attention_heads, dropout_p):
        super().__init__()
        self.query_projection = torch.nn.linear(hidden_size, hidden_size)
        self.key_projection = torch.nn.linear(hidden_size, hidden_size)
        self.value_projection = torch.nn.linear(hidden_size, hidden_size)
        self.dropout = torch.nn.dropout(dropout_p)
 
    def forward(self, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = self.dropout(softmax_qk)
        output = dropout_qk.matmul(value)
        return output
        
# Initializing the model
hidden_size = 64
num_attention_heads = 8
dropout_p = 0.1
m = Model(hidden_size, num_attention_heads, dropout_p)

# Inputs to the model
query = torch.randn(hidden_size, hidden_size)
key = torch.randn(hidden_size, hidden_size)
value = torch.randn(hidden_size, hidden_size)
