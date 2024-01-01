
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 32, 1)

        self.scale_factor = torch.nn.Parameter(torch.tensor(1.0))
        self.dropout_p = 0.1
 
    def forward(self, q, k, v):
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.mul(self.scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.1)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(2, 4, 32)
k = torch.randn(2, 4, 32)
v = torch.randn(2, 4, 32)
