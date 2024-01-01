
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout_p)
 
    def forward(self, query, key, value, scale, dropout_p):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.mul(scale)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = self.dropout(softmax_qk)
        return dropout_qk.matmul(value)
 
# Initializing the model
m = Model()
 
# Inputs to the model
query = torch.randn(1, 32, 256)
key = torch.randn(1, 64, 256)
value = torch.randn(1, 64, 256)
scale = torch.tensor(math.sqrt(1. / 256))
dropout_p = 0.3
