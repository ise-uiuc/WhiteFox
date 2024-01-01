
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, q, k, v, mask):
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(np.power(key.size(-1), -0.5))
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = F.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(v)
        return output
 
# Initializing the model
m = Model()
 
# Inputs to the model
q = torch.randn(1, 8, 64, 64)
k = torch.randn(1, 8, 128, 128)
v = torch.randn(1, 8, 128, 128)
mask = torch.ones_like(q[:,:,0,0]).bool()
