
 class DotProductAttention(torch.nn.Module):
    def __init__(self, inv_scale_factor: float = 1.0, dropout_p: float = 0.0):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout_p)
        self.dropout_p = dropout_p
        self.inv_scale_factor = inv_scale_factor
 
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor)
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(self.inv_scale_factor)
        attn = self.softmax(scaled_qk)
        return self.dropout(attn).matmul(value)

# Initializing the model
m = DotProductAttention()

# Inputs to the model
query = torch.randn(1, 8, 100)
key = torch.randn(1, 8, 100)
value = torch.randn(1, 8, 100)
