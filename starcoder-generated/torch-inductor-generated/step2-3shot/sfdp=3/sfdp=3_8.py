
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(0.1)
 
    def forward(self, query, key, value, scale=1.0):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.mul(scale)

        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = self.dropout(softmax_qk)

        return dropout_qk.matmul(value)

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 16, 8, 8)
key = torch.randn(1, 8, 16, 16)
value = torch.randn(1, 8, 16, 16)
