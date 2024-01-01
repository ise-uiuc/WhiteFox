
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = torch.nn.Softmax(dim=-1)
 
    def forward(self, q, k, v, scale_factor, dropout_p):
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = self.softmax(scaled_qk)
        output = softmax_qk.matmul(v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(16, 32, 64, 64)
k = torch.randn(16, 32, 64, 64)
v = torch.randn(16, 32, 64, 64)
scale_factor = torch.randn(16, 32, 1, 1)
dropout_p = 0.5
