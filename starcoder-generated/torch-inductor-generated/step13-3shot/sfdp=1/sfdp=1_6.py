
class Model(torch.nn.Module):
    def __init__(self, num_heads, qkv_dim, dropout_p):
        super().__init__()
        self.qkv = torch.nn.Linear(qkv_dim, qkv_dim * 3)
        self.num_heads = num_heads
        self.dropout_p = dropout_p
 
    def forward(self, qkv):
        qkv = self.qkv(qkv)
        query, key, value = torch.chunk(qkv, 3, dim=-1)
        num_qkv = query.shape[-2]
        inv_scale_factor = 1.0 / (self.num_heads ** 0.5)
        scaled_qk = torch.matmul(query, key.transpose(-2, -1)).div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = torch.matmul(dropout_qk, value)
        return output

# Initializing the model
m = Model(num_heads=64, qkv_dim=64, dropout_p=0.05)

# Inputs to the model
x1 = torch.randn(1, 64, 64)
