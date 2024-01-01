
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, Q, K, V, dropout_p):
        k = K.transpose(-2, -1)
        inv_scale_factor = float(K.size(-1)) ** -0.5
        qk = torch.matmul(Q, k)
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        drop_softmax_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        drop_softmax_qk_V = torch.matmul(drop_softmax_qk, V)
        return drop_softmax_qk_V
        

# Initializing the model
dropout_p = 0.1 
m = Model()

# Inputs to the model
n = 64
Q = torch.randn(2, n, 35, 35)
K = torch.randn(2, n, 35, 35)
V = torch.randn(2, n, 35, 35)
