
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, Q, K, V, dropout_p = 0.5, scale_factor = 128.0):
        qk = torch.matmul(Q, K.transpose(-2, -1))
        inv_scale_factor = 1 / scale_factor
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        drop_softmax_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = drop_softmax_qk.matmul(V)
        return output

# Initializing the model
m = Model()

# Inputs to the model
Q = torch.randn(1, 32, 512, 128)
K = torch.randn(1, 32, 512, 128)
V = torch.randn(1, 32, 512, 128)
