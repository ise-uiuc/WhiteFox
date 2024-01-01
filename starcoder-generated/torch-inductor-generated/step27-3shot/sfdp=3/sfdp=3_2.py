
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = Linear(query_size, key_size)
 
    def forward(self, query, key, scale_factor, dropout_p):
        q = self.proj(query)
        k = self.proj(key)
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 16, query_size)
key = torch.randn(1, 16, key_size)
scale_factor = torch.rand(1, 16)
dropout_p = 0.5
value = torch.randn(1, 16, value_size)
