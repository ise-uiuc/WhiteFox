
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q = torch.nn.Linear(1000, 500)
        self.k = torch.nn.Linear(1000, 500)
        self.v = torch.nn.Linear(1000, 500)
 
    def forward(self, x1, x2, x3, x4, x5):
        q = self.q(x1)
        k = self.k(x2)
        v = self.v(x3)

        qk = torch.matmul(q, k.transpose(-2, -1))
        inv_scale_factor = 5
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_p = 0.5
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 1000)
x2 = torch.randn(1, 1000)
x3 = torch.randn(1, 1000)
x4 = torch.randn(1, 1000)
x5 = torch.randn(1, 1000)
