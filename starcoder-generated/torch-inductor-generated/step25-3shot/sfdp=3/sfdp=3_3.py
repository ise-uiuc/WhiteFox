
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale_factor = np.sqrt(1.0 / 8)
 
    def forward(self, Q, K, V, dropout_p):
        qk = torch.matmul(Q, K.transpose(-2, -1))
        v1 = qk.mul(self.scale_factor)
        softmax_qk = v1.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(V)
        return output
 
# Initializing the model
m = Model()

Q = torch.randn(1, 1, 1, 8)
K = torch.randn(1, 1, 8, 8)
V = torch.randn(1, 1, 8, 8)
__dropout_p__ = 0.0
