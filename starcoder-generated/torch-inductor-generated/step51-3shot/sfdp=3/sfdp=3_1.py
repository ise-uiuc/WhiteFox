
class Scores(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
 
    def forward(self, x1, x2):
        Q = x1
        K = x1 if x2 is None else x2
        qk = torch.matmul(Q, torch.transpose(K, -2, -1))
        scale_factor = np.power(np.float32(K.size(self.dim)) \
        , -0.5)
        softmax_qk = torch.softmax(qk * scale_factor, dim=-1)
        dropout_qk = softmax_qk
        output = torch.matmul(dropout_qk, V)
        return output

# Initializing the model
scores = Scores(dim=1)

# Inputs to the model
x1 = torch.randn(1, 3, 64)
