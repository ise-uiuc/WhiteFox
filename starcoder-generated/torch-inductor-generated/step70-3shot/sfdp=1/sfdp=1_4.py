
class Model(nn.Module):

    def __init__(self, d_model, query, key, value):
        super().__init__()
        self.scale_factor = math.sqrt(query.shape[-1])
        self.dropout_p = 0.5
        self.dropout = nn.Dropout(self.dropout_p)
        self.softmax = nn.Softmax(dim=-1)
        self.matmul_query_key = torch.matmul(query, key.transpose(-2, -1))
 
    def forward(self, x):
        qk = self.matmul_query_key
        scaled_qk = qk.div(self.scale_factor)
        softmax_qk = self.softmax(scaled_qk)
        dropout_qk = self.dropout(softmax_qk).matmul(value)
        return dropout_qk

# Initializing a query, key and value tensor
d_model = 8
query = torch.randn(1, 1, 7)
key = torch.randn(1, 1, 10)
value = torch.randn(1, 1, 10)

# Inputs to the model
x = torch.randn(1, 1, 7)
