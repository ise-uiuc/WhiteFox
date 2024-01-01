
class Model(torch.nn.Module):
    def __init__(self, dim, dropout_p=0.2):
        super().__init__()
        self.dim = dim
        self.dropout_p = dropout_p
        self.softmax = nn.Softmax(dim=dim)
        self.dropout = nn.Dropout(dropout_p)
 
    def forward(self, query, key, value, inv_scale_factor=0.5):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = self.softmax(scaled_qk)
        dropout_qk = self.dropout(softmax_qk)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model(2048)

# Inputs to the model
query = torch.randn(1, 10, 2048)
key = torch.randn(1, 25, 2048)
value = torch.randn(1, 25, 2048)
inv_scale_factor = 0.5
