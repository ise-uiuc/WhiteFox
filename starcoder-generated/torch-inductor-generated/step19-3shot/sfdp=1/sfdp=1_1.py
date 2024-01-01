
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout_p)
 
    def forward(self, q, k, v):
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = xk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = self.dropout(softmax_qk)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
inv_scale_factor = 256.0
dropout_p = 0.3
m = Model()

# Inputs to the model
k = torch.randn(1, 32, 5, 5)
q = torch.randn(1, 32, 5, 5)
v = torch.randn(1, 32, 5, 5)
