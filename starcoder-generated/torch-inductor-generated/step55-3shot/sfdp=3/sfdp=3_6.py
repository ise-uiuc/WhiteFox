
class Model(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
 
    def forward(self, x1):
        k = x1
        q = x1
        v = x1
        scaled_qk = torch.matmul(q, k.transpose(-2, -1)) * self.dim ** -0.5
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.1)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model(dim=5)

# Inputs to the model
x1 = torch.randn(4, 32, 5)
