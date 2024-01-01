
class Model(torch.nn.Module):
    def __init__(self, d_k):
        super().__init__()
        self.w = torch.nn.Linear(d_k, d_k)
 
    def forward(self, query, key, value, dropout_p):
        qk = torch.matmul(query, key.transpose(-2, -1))
        inv_scale_factor = torch.rsqrt(torch.tensor(1/self.w.in_features))
        scaled_qk = qk * inv_scale_factor
        output = scaled_qk.softmax(dim=-1).matmul(value)
        output = torch.nn.functional.dropout(output, p=dropout_p)
        return output

# Initializing the model
d_k = 512
m = Model(d_k)

# Inputs to the model
query = torch.randn(1, 64, d_k)
key = torch.randn(1, 2, d_k)
value = torch.randn(1, 2, d_k)
dropout_p = 0.125
