
class Model(torch.nn.Module):
    def __init__(self, in_features, out_features, dropout_p, div_val):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout_p)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.linear_in = torch.nn.Linear(in_features, out_features)
        self.div_val = div_val
 
    def forward(self, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(self.div_val)
        softmax_qk = self.softmax(scaled_qk)
        dropout_qk = self.dropout(softmax_qk)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model(20, 4, 0.3, 32)

# Initializing the input tensor
query = torch.randn(5, 16, 20)
key = torch.randn(5, 8, 20)
value = torch.randn(5, 8, 4)

# Getting the output of the model
