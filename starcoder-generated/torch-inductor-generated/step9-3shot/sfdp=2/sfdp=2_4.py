
class Model(torch.nn.Module):
    def __init__(self, query_size, key_size, value_size):
        super().__init__()
        self.query = torch.nn.Parameter(torch.empty(query_size))
        self.key = torch.nn.Parameter(torch.empty(key_size))
        self.value = torch.nn.Parameter(torch.empty(value_size))
 
    def forward(self, x):
        qk = torch.matmul(x, self.query.unsqueeze(-1))
        qk = torch.matmul(qk, self.key.unsqueeze(-1).transpose(-2, -1))
        inv_scale_factor = torch.rsqrt(torch.tensor(self.key.shape[-1], dtype=torch.float32)).to(x.device)
        scaled_qk = qk * inv_scale_factor
        softmax_qk = torch.nn.functional.softmax(scaled_qk, dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        return dropout_qk.matmul(self.value)

# Initializing the model
m = Model(query_size, key_size, value_size)

# Inputs to the model
x = torch.randn(batch_size, query_size, seq_length)
