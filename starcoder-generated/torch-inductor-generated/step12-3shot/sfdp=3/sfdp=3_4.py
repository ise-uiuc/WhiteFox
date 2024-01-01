
class Model(nn.Module):
    def __init__(self, dropout_p):
        super().__init__()
        self.dropout_p = dropout_p
 
    def forward(self, q, k, v, scale_factor):
        qk = F.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = F.softmax(scaled_qk, dim=-1)
        dropout_qk = F.dropout(softmax_qk, p=self.dropout_p)
        output = F.matmul(dropout_qk, v)
        return output
    

# Initializing the model
dropout_p = 0.2
m = Model(dropout_p)

# Inputs to the model
q = torch.randn(1, 8, 64)
k = torch.randn(1, 8, 64)
v = torch.randn(1, 8, 64)
scale_factor = 1 / math.sqrt(k.size(-1))

