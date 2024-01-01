
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(8192)
 
    def forward(self, q, k, v, dropout_p=8192):
        scale_factor = (1.0 / math.sqrt(q.size(-1)))
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = self.dropout(softmax_qk)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(1, 100, 512)
k = torch.randn(1, 100, 384)
v = torch.randn(1, 100, 8192)

