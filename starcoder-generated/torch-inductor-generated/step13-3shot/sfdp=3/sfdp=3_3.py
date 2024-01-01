
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, query, key, value, scale_factor=None, dropout_p=None):
        qk = torch.matmul(query, key.transpose(-2, -1)).mul(scale_factor)
        softmax_qk = qk.softmax(dim=-1).mul(dropout_p)
        dropout_qk = softmax_qk.matmul(value)
        return dropout_qk

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 1, 3, 8)
key = torch.randn(1, 8, 20, 4)
value = torch.randn(1, 8, 20, 8)
dropout_p = 0.5
