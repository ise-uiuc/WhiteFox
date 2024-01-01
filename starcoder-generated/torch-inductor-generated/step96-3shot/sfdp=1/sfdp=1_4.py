
class Model(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.scale_factor = torch.nn.Parameter(torch.tensor([[1.0]]))
        self.dropout_p = 0.0
 
    def forward(self, q, k, v):
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(self.scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
d = 4
m = Model(d)

# Inputs to the model
q = torch.randn(1, 16, d)
k = torch.randn(1, 256, d)
v = torch.randn(1, 256, d)
