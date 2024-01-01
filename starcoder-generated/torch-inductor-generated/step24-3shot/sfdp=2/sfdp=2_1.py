
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = torch.nn.Softmax(dim=-1)
        self.matmul1 = torch.nn.Linear(5, 4)
        self.matmul2 = torch.nn.Linear(4, 3)
        self.matmul3 = torch.nn.Linear(4, 2)
 
    def forward(self, q, k, v, inv_scale_factor, dropout_p):
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = self.softmax(scaled_qk)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        v1 = self.matmul1(dropout_qk)
        v2 = torch.transpose(v1, -2, -1)
        v3 = self.matmul2(torch.transpose(v2, -1, -2))
        v4 = self.matmul3(v3)
        output = torch.transpose(v4, -2, -1)
        return output

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(2, 4, 5)
k = torch.randn(2, 5, 3)
v = torch.randn(2, 5, 2)
__inv_scale_factor__ = 2**0.5
__dropout_p__ = 0.0
