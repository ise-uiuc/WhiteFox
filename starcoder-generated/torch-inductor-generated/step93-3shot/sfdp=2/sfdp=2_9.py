
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(0.1)
 
    def forward(self, q, k, v, scale_factor, dropout_p):
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        drop_qk = self.dropout(softmax_qk)
        return torch.matmul(drop_qk, v)

# Initializing the model
m = Model()

# Inputs to the model
scale_factor = 0.2
dropout_p = 0.1
q = torch.randn(3, 5, 4)
k = torch.randn(3, 6, 4)
v = torch.randn(3, 6, 5)
