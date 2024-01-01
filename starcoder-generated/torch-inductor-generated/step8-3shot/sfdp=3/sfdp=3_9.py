
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(0.02)
        self.dropout_p = 0.01
        self.scale_factor = 1 / math.sqrt(1024)
        self.num_heads = 2
        
    def forward(self, q, k, v):
        qk = q.matmul(k.transpose(-2, -1))
        scaled_qk = qk.mul(self.scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = self.dropout(softmax_qk)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(2, 3, 64, 1024)
k = torch.randn(2, 3, 64, 1024)
v = torch.randn(2, 3, 64, 1024)
