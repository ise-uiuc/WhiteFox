
class Model(torch.nn.Module):
    def __init__(self, dropout_p, scale_factor):
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout_p)
        self.scale_factor = scale_factor
 
    def forward(self, q, k, v):
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.mul(self.scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = self.dropout(softmax_qk)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
dropout_p = 0.1
scale_factor = 0.125
m = Model(dropout_p, scale_factor)

# Inputs to the model
q = torch.randn(7, 16, 512)
k = torch.randn(7, 32, 512)
v = torch.randn(7, 32, 512)
