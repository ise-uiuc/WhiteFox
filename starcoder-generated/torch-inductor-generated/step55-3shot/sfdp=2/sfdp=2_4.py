
class Model(torch.nn.Module):
    def __init__(self, query_dim, key_dim, num_values, dropout_p):
        super().__init__()
        self.dropout_p = dropout_p
        self.wq = torch.nn.Linear(query_dim, query_dim, bias=False)
        self.wk = torch.nn.Linear(key_dim, query_dim, bias=False)
        self.wv = torch.nn.Linear(num_values, query_dim, bias=False)
 
    def forward(self, query, key, value, mask, scale_factor, inv_scale_factor):
        q = self.wq(query)
        k = self.wk(key)
        v = self.wv(value)
        attn_qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_attn_qk = attn_qk.div(scale_factor)
        softmax_attn_qk = scaled_attn_qk.softmax(dim=-1)
        dropout_attn_qk = torch.nn.functional.dropout(softmax_attn_qk, p=self.dropout_p)
        attn = dropout_attn_qk.matmul(v)
        return attn, torch.empty(0, device=attn.device, dtype=attn.dtype)

# Initializing the model
m = Model(32, 256, 32, 0.25)

# Inputs to the model
query = torch.randn(128, 12, 32)
key = torch.randn(128, 6, 256)
value = torch.randn(128, 6, 32)
mask = torch.zeros(128, 6, 6, dtype=torch.bool)
scale_factor = 10
inv_scale_factor = 1 / scale_factor
