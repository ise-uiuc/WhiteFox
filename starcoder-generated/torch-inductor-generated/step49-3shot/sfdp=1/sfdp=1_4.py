
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout_qk = torch.nn.Dropout(p=0.4)
 
    def forward(self, q, k, v):
        qk = torch.matmul(q, k.transpose(-2, -1))
        inv_scale_factor = k.size(-2)**0.25
        s_qk = qk.div(inv_scale_factor)
        softmax_qk = s_qk.softmax(dim=-1)
        d_qk = self.dropout_qk(softmax_qk)
        return d_qk.matmul(v)
 
# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(1, 1, 20)
k = torch.randn(1, 1, 20)
v = torch.randn(1, 20, 20)
