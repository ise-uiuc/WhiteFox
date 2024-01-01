
class Model(torch.nn.Module):
    def __init__(self, query_features, key_features, num_heads, dropout_p):
        super().__init__()
        self.q_linear = torch.nn.Linear(query_features, key_features)
        self.k_linear = torch.nn.Linear(key_features, key_features)
        self.v_linear = torch.nn.Linear(key_features, key_features)
        self.dropout = torch.nn.Dropout(p=dropout_p)
        self.num_heads = num_heads
 
    def forward(self, query, key, value, inv_scale_factor):
        q = self.q_linear(query).chunk(self.num_heads, dim=-1)
        k = self.k_linear(key).chunk(self.num_heads, dim=-1)
        v = self.v_linear(value).chunk(self.num_heads, dim=-1)
        scaled_qk = torch.cat([qk.matmul(k.transpose(-2, -1)).div(inv_scale_factor) for qk in q], dim=-2)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = self.dropout(softmax_qk)
        output = torch.cat([dropout_qk.matmul(value) for value in v], dim=-2)
        return output

# Initializing the model
query = torch.randn(5, 4, 10)
key = torch.randn(6, 4, 12)
value = torch.randn(6, 4, 14)
inv_scale_factor = 0.5
dropout_p = 0.5
num_heads = 3
m = Model(query.size(-1), key.size(-1), num_heads, dropout_p)
num_q_splits = query.size(1) // num_heads
num_k_splits = key.size(1) // num_heads
num_v_splits = value.size(1) // num_heads

# Inputs to the model
