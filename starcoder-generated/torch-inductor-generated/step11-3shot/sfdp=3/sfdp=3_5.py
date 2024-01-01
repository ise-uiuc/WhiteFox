
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(p = 0.0)
 
    def forward(self, q, k, v, scale_factor, dropout_p):
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = self.dropout(softmax_qk).mul(dropout_p)
        output = dropout_qk.matmul(v)
        return output
 
# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(2, 6, 5)
k = torch.randn(2, 4, 6)
v = torch.randn(2, 4, 5)
scale_factor = torch.randn(12, 5)
dropout_p = 0.0
