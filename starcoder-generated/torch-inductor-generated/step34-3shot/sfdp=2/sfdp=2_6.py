
class Model(torch.nn.Module):
    def __init__(self, seq_len: int = 32, query_feature: int = 8, key_feature: int = 16):
        super().__init__()
        self.query = torch.nn.Parameter(torch.randn(1, seq_len, query_feature))
        self.key = torch.nn.Parameter(torch.randn(1, seq_len, key_feature))
        self.inv_scale_factor = torch.nn.Parameter(torch.randn(1))
        self.dropout = torch.nn.Dropout(0.1)
 
    def forward(self, value, dropout_p_):
        qk = torch.matmul(self.query, self.key.transpose(-2, -1))
        scaled_qk = qk.div(self.inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = self.dropout(softmax_qk)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model(seq_len=32, query_feature=8, key_feature=16)

# Inputs to the model
value = torch.randn(1, seq_len=32, value_feature=16)
dropout_p_ = 0.1
