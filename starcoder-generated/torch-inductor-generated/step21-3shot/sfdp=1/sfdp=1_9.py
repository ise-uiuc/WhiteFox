
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, inv_scale_factor=1.0, dropout_p=0.0):
        query = self.query_proj(q)
        key = self.key_proj(k)
        value = self.value_proj(v)
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        return dropout_qk.matmul(value)

# Initializing the model
m = Model()

# Inputs to the model
b, c, h, w = 8, 32, 32, 32
q = torch.randn(1, b, c, h // 4, w // 4)
k = torch.randn(1, b, c, h // 4, w // 4)
v = torch.randn(1, b, c, h // 4, w // 4)
