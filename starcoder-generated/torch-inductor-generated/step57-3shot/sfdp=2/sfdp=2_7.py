
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_dim = 32
        self.head_dim = self.embed_dim // 4
        self.w_qkv = torch.nn.Linear(self.embed_dim, 3 * self.head_dim, bias=False)
 
    def forward(self, query, key, value, inv_scale_factor, dropout_p):
        qkv = self.w_qkv(query)
        q, k, v = qkv.reshape(query.size(0), query.size(1), 3, self.head_dim).transpose(1, 2)
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = torch.matmul(dropout_qk, v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(8, 4, 32)
key = torch.randn(8, 12, 32)
value = torch.randn(8, 12, 32)
inv_scale_factor = 1.0
dropout_p = 0.1
