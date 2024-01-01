
class ScaledDotProductAttention(torch.nn.Module):
    def __init__(self, query_features, key_features, value_features, dropout_p):
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout_p)
        self.softmax = torch.nn.Softmax(dim=-1)
 
    def forward(self, query, key, value, scale_factor=math.sqrt(dim_k)):
        scaled_qk = torch.matmul(query, key.transpose(-2, -1)) * scale_factor
        softmax_qk = self.softmax(scaled_qk)
        dropout_qk = self.dropout(softmax_qk)
        output = torch.matmul(dropout_qk, value)
        return output

# Initializing the model
m = ScaledDotProductAttention(query_features=128, key_features=128, value_features=128, dropout_p=0.1)

# Inputs to the model
query = torch.randn(1, 5, 128)
key = torch.randn(1, 6, 128)
value = torch.randn(1, 6, 128)
