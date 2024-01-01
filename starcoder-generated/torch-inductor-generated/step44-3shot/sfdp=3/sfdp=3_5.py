
class DotProductAttention(nn.Module):
    def __init__(self, dropout_p=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)
     
    def forward(self, query, key, value, scale_factor=None):
        qk = query.matmul(key.transpose(-1, -2))
        if scale_factor:
            qk = qk * scale_factor
        softmax_qk = qk.softmax(dim=-1)
        dropout_qk = self.dropout(softmax_qk)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m1 = DotProductAttention(dropout_p=0.1)

# Inputs to the model
q = torch.randn(16, 1, 512)
k = torch.randn(16, 1, 512)
v = torch.randn(16, 10, 512)
