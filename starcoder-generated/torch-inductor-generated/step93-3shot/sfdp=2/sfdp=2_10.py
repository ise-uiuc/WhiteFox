
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_q = torch.nn.Linear(d_model, d_model)
        self.linear_k = torch.nn.Linear(d_model, d_model)
        self.linear_v = torch.nn.Linear(d_model, d_model)
 
    def forward(self, q, k, v):
        qt = self.linear_q(q)
        kt = self.linear_k(k)
        vt = self.linear_v(v)
        qk = torch.matmul(qt, kt.transpose(-2, -1))
        inv_scale_factor = torch.sqrt(torch.tensor(float(d_model)).float())
        dropout_p = 0.4
        v3 = qk.div(inv_scale_factor)
        v4 = v3.softmax(dim=-1)
        v5 = torch.nn.functional.dropout(v4, p=dropout_p)
        output = v5.matmul(vt)
        return output

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(1, 16, 32)
k = torch.randn(1, 16, 48)
v = torch.randn(1, 16, 64)
