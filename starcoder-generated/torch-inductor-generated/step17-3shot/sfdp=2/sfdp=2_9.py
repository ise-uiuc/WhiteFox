
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, scale_factor, dropout_p): 
        qk = torch.matmul(q, k.transpose(-2, -1))
        v_ = v.permute(0,1,3,2)
        qk_ = qk.permute(0,2,3,1)
        qkv = torch.matmul(qk_, v_)
        qkv_ = qkv.permute(0,3,1,2)
        return qkv_

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(2, 3, 8, 8)
k = torch.randn(2, 3, 8, 8)
v = torch.randn(2, 3, 8, 8)
scale_factor = 0.1
dropout_p = 0.5
