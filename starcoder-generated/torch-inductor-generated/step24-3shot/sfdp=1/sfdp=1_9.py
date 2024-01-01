
class ScaledDotProductAttention(torch.nn.Module):
    def __init__(self, dropout_p):
        super().__init__()
        self.dropout_p = dropout_p
 
    def forward(self, query, key, value, inv_scale_factor):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(value)
        return output
 
class Model(torch.nn.Module):
    def __init__(self, dropout_p):
        super().__init__()
        self.transformer_block = torch.nn.TransformerEncoderLayer(d_model=64, nhead=8)
        self.attention = ScaledDotProductAttention(dropout_p)
 
    def forward(self, x1, x2, x3, x4):
        v1 = self.transformer_block(x1)
        v2 = x2.repeat(1, 8, 1, 1)
        v3 = x3.repeat(1, 8, 1, 1)
        v4 = self.attention(query=v1, key=v2, value=v3, inv_scale_factor=1 / 64**0.5)
        v5 = v4 + x4
        return v5

# Initializing the model
m = Model(dropout_p=0.1)

# Inputs to the model
x1 = torch.randn(1, 64, 64, 64)
x2 = torch.randn(1, 1, 64, 64)
x3 = torch.randn(1, 1, 64, 64)
x4 = torch.randn(1, 8, 64, 64)
