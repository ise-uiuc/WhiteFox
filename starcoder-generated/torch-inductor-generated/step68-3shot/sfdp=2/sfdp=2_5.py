
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q = torch.nn.Linear(512, 512)
        self.k = torch.nn.Linear(512, 512)
 
    def forward(self, q, k):
        q = self.q(q)
        k = self.k(k)
        qk = torch.matmul(q, k.transpose(-2, -1))
        inv_scale_factor = 1 / math.sqrt(512)
        softmax_qk = qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.3)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(1, 100, 512)
k = torch.randn(1, 900, 512)
v = torch.randn(1, 900, 512)
