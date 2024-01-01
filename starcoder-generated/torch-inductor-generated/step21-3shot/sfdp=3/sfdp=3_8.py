
class Model(torch.nn.Module):
    def __init__(self, key_size, dropout_p):
        super().__init__()
        self.scale_factor = 1.0 / math.sqrt(key_size)
        self.dropout = torch.nn.Dropout(dropout_p)
 
    def forward(self, q, k, v):
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = self.scale_factor * qk
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = self.dropout(softmax_qk)
        return dropout_qk.matmul(v)

# Inputs to the model
q = torch.randn(5, 4, 200)
k = torch.randn(5, 6, 200)
v = torch.randn(5, 6, 100)
