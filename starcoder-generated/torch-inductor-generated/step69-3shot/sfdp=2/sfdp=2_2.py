
class ScaledDotProductAttention(torch.nn.Module):
    def __init__(self, query, key, value, dropout_p=0.1):
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout_p)
        self.softmax1 = torch.nn.Softmax(dim=-1)
        self.softmax2 = torch.nn.Softmax(dim=-2)
        self.matmatmul = torch.matmul
 
    def forward(self, query, key, inv_scale_factor, value):
        qk = self.matmatmul(query, key.transpose(-2, -1))
        scaled_qk = qk / inv_scale_factor.unsqueeze(-1)
        softmax_qk = self.softmax1(scaled_qk)
        dropout_qk = self.dropout(softmax_qk)
        output = torch.matmul(dropout_qk, value)
        return output

# Initializing the model
m = ScaledDotProductAttention(query, key, value)

# Inputs to the model
inv_scale_factor = torch.randn(16, 1, 1)
dropout_p=0.1
