
class Model(torch.nn.Module):
    def __init__(self, query, key, value, scale_factor, dropout_p):
        super().__init__()
 
 
    def forward(self, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = torch.nn.functional.softmax(scaled_qk, dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        return torch.matmul(dropout_qk, value)

# Initializing the model
m = Model(
        query=query, 
        key=key, 
        value=value, 
        scale_factor=scale_factor, 
        dropout_p=dropout_p,
        )

# Inputs to the model
query = torch.randn(batch_size, seq_length, seq_length, hidden_size)
key = torch.randn(batch_size, seq_length, seq_length, hidden_size)
value = torch.randn(batch_size, seq_length, seq_length, hidden_size)
