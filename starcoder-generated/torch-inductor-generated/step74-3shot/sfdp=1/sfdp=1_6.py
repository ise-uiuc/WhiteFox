
class Model(torch.nn.Module):
    def __init__(self, dim, inv_scale_factor):
        super().__init__()
        self.query = torch.nn.Linear(dim, dim)
        self.key = torch.nn.Linear(dim, dim)
        self.value = torch.nn.Linear(dim, dim)
        self.inv_scale_factor = inv_scale_factor
 
    def forward(self, q, k, v, p):
        q, k, v = self.query(q), self.key(k), self.value(v)
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(self.inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=p)
        output = dropout_qk.matmul(v)
        return output
 
# Intializing the model
m = Model(512, 16)
 
# Inputs to the model
q = torch.randn(1, 256, 512)
k = torch.randn(1, 256, 512)
v = torch.randn(1, 256, 512)
p = 0.3
 
