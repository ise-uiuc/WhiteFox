
class Model(torch.nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()
        self.head_dim = hidden_size // 4
        self.scale_factor = math.sqrt(self.head_dim)
        self.dropout = torch.nn.Dropout(dropout_p)
 
    def forward(self, query, key, value, dropout_p):
        qk = query.matmul(key.transpose(-2, -1))
        inv_scale_factor = torch.tensor(1.0 / self.scale_factor, device=qk.device)
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = torch.nn.functional.softmax(scaled_qk, dim=-1)
        dropout_qk = self.dropout(softmax_qk)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model(hidden_size=64)
num_qkv = 8
# Inputs to the model
query = torch.randn(1, num_qkv, hidden_size)
key = torch.randn(1, num_qkv, hidden_size)
value = torch.randn(1, num_qkv, hidden_size)
dropout_p = 0.2
