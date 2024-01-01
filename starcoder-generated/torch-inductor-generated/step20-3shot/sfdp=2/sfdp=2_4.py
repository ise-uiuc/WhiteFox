
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, Q, K, V, dropout_p, inv_scale_factor):
        Q = Q.transpose(0, 1)
        K = K.transpose(0, 1)
        Kt = K.transpose(-2, -1)
        QK = torch.matmul(Q, Kt)
        scaled_QK = QK.div(inv_scale_factor)
        softmax_QK = F.softmax(scaled_QK, dim=-1)
        dropout_QK = F.dropout(softmax_QK, p=dropout_p)
        output = torch.matmul(dropout_QK, V.transpose(0, 1))
        return output

# Initializing the model
m = Model()

# Inputs to the model
Q = torch.randn(16, 1, 1024)
K = torch.randn(16, 1, 1024)
V = torch.randn(16, 1, 1024)
dropout_p = 0.5
