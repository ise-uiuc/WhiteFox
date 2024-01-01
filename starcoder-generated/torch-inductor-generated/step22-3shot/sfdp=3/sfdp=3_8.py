
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q = torch.nn.Parameter(torch.randn(1, 64, 1))
        self.k = torch.nn.Parameter(torch.randn(1, 64, 1))
 
    def forward(self, v):
        qk = torch.matmul(self.q, self.k.transpose(-2, -1))
        scale_factor = 2**(-1/64)
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.1)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
v = torch.randn(1, 64, 64)
