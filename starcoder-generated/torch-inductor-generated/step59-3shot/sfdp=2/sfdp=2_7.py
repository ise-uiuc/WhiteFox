
class Model(torch.nn.Module):
    def __init__(self, D=32, H=8, N=64, dropout_p=0.25):
        super().__init__()
        self.dropout_p = dropout_p
        self.Q = torch.nn.Linear(D, H)
        self.K = torch.nn.Linear(D, H)
        self.V = torch.nn.Linear(D, H)
 
    def forward(self, q, k, v):
        Q = self.Q(q)
        K = self.K(k)
        V = self.V(v)
        Q /= float(Q.shape[-1]) ** 0.5
        K /= float(K.shape[-1]) ** 0.5
        softmax_qk = torch.matmul(Q, K.transpose(-2, -1))
        inv_scale_factors = torch.rsqrt((Q ** 2).sum(-1, keepdim=True))
        scaled_softmax_qk = softmax_qk * inv_scale_factor
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(V)
        return output

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(1, 32, 16)
k = torch.randn(1, 32, 16)
v = torch.randn(1, 32, 16)
