
class Model(torch.nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, scale_factor, dropout_p):
        super().__init__()
        self.scale_factor = scale_factor
        self.softmax = torch.nn.Softmax(dim=-1)
        self.dropout = torch.nn.Dropout(dropout_p)
 
    def forward(self, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.mul(self.scale_factor)
        softmax_qk = self.softmax(scaled_qk)
        output = self.dropout(softmax_qk).matmul(value)
        return output

# Initializing the model
m = Model(512, 512, 512, 10.0, 0.1)

# Inputs to the model
query = torch.randn(32, 512, 32, 16)
key = torch.randn(32, 512, 16, 16)
value = torch.randn(32, 512, 16, 32)
