
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax_qk = torch.nn.Softmax(dim=-1)
 
    def forward(self, x1, x2, scale_factor, dropout_p):
        v1 = torch.matmul(x1, x2.transpose(-2, -1))
        scaled_qk = v1.mul(scale_factor)
        softmax_qk = self.softmax_qk(scaled_qk).mul(dropout_p).add(1 - dropout_p)
        output = softmax_qk.matmul(x2)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 12, 30, 64)
x2 = torch.randn(30, 64)
scale_factor = 1 / math.sqrt(64)
dropout_p = 0.1
