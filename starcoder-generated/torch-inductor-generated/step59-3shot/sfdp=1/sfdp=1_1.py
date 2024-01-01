
class ScaledDotProductAttention(torch.nn.Module):
    def __init__(self, dropout, d):
        super().__init__()
        self.dropout = dropout
        self.scale_factor = torch.sqrt(torch.tensor(d, dtype=torch.float))
    
    def forward(self, q, k, v):
        scaled_qk = torch.matmul(q, k.transpose(-2, -1)) / self.scale_factor
        softmax_qk = F.softmax(scaled_qk, dim=-1)
        dropout_qk = F.dropout(softmax_qk, p=self.dropout, training=self.training)
        output = torch.matmul(dropout_qk, v)
        return output

# Initializing the model
d = 1
dropout = 0.1
m = ScaledDotProductAttention(dropout, d)

# Inputs to the model
query = torch.randn(1, 2, d)
key = torch.randn(1, 4, d) * 0.02
value = torch.randn(1, 4, d) * 0.02
