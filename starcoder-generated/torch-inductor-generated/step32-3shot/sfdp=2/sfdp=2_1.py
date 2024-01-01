
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.softm = torch.nn.Softmax(dim=-1)
        self.dropout = torch.nn.Dropout(dropout_p)
 
    def forward(self, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = self.softm(scaled_qk)
        dropout_qk = self.dropout(softmax_qk)
        return dropout_qk.matmul(value)

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, num_heads, seq_length, depth)
key = torch.randn(1, num_heads, seq_length, depth)
value = torch.randn(1, num_heads, seq_length, depth)
