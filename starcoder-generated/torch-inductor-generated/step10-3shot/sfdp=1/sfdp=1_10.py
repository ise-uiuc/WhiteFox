
class Model(torch.nn.Module):
    def __init__(self, q, k, v):
        super().__init__()
        self.m = nn.Linear(q, k)
 
    def forward(self, query, key, value):
        qk = self.m(query).matmul(torch.transpose(key, -2, -1))
        inv_scale_factor = math.sqrt(key.shape[-1] - 1)
        scaled_qk = qk / inv_scale_factor
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.2)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
in_q = 32
in_k = 64
in_v = 64
m = Model(in_q, in_k, in_v)

# Inputs to the model
query = torch.randn(16, 8, in_q)
key = torch.randn(16, 8, in_k)
value = torch.randn(16, 8, in_v)
