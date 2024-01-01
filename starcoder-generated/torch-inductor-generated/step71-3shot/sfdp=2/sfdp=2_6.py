
class Model(torch.nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.dropout_p = p
        self.d_model = d_model
        self.w_q = torch.nn.Linear(100, d_model)
        self.w_k = torch.nn.Linear(100, d_model)
        self.w_v = torch.nn.Linear(100, d_model)
        self.w_o = torch.nn.Linear(d_model, 100)
 
    def forward(self, x1, x2):
        q = self.w_q(x1).reshape(x1.size(0), self.d_model, 1)
        k = self.w_k(x2).reshape(x2.size(0), self.d_model, 1).transpose(-2, -1)
        v = self.w_v(x2).reshape(x2.size(0), self.d_model, 1)
        qk = q.matmul(k).reshape(x1.size(0) * self.d_model, x1.size(1)).reshape(x1.size(0), self.d_model, x1.size(1)).transpose(-2, -1)
        inv_scale_factor = torch.sqrt(torch.Tensor([x2.size(2)])).repeat(x2.size(0)).reshape(x2.size(0), x2.size(2))
        dropout_qk = torch.nn.functional.dropout(qk.div(inv_scale_factor), p=self.dropout_p)
        output = torch.tanh(dropout_qk.matmul(v).reshape(x1.size(0) * self.d_model, x1.size(1)).reshape(x1.size(0), self.d_model, x1.size(1)))
        output = self.w_o(output)
        return output.reshape(x1.size(0), 100)

# Initializing model
d_model = 100
p = 0.1
m = Model(d_model)

# Input to the model
x1 = torch.randn(1, 5, 100)
x2 = torch.randn(1, 5, 100, 6)
