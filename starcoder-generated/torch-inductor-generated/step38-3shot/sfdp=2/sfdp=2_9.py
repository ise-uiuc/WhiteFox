
class Model(torch.nn.Module):
    def __init__(self, query_size, key_size, value_size, scale_factor):
        super().__init__()
        self.dropout = torch.nn.Dropout()
        self.softmax_qk = torch.nn.Softmax(dim = -1)
        self.inv_scale_factor = 1. / float(scale_factor)
 
    def forward(self, query, key, value, dropout_p):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(self.inv_scale_factor)
        softmax_qk = self.softmax_qk(scaled_qk)
        dropout_qk = self.dropout(softmax_qk)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model(query_size=3, key_size=5, value_size=5, scale_factor=10000)

# Inputs to the model
query = torch.randn(2, 3)
key = torch.randn(2, 5)
value = torch.randn(2, 5)
dropout_p = 0.1
