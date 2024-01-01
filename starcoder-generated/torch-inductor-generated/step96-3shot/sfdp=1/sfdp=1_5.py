
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q = torch.nn.Linear(128, 128)
        self.k = torch.nn.Linear(128, 128)
        self.v = torch.nn.Linear(128, 128)
 
    def forward(self, x1):
        x1 = x1.transpose(0, 1)
        q = self.q(x1)
        k = self.k(x1)
        v = self.v(x1)
        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        qk = torch.matmul(q, k.transpose(-2, -1))
        inv_scale_factor = torch.tensor(1 / math.sqrt(128), dtype=torch.float32)
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.2)
        output = dropout_qk.matmul(v)
        return output, (softmax_qk, x1)

# Initializing the model
m = Model()

# Input to the model
x1 = torch.randn(128, 12, 128)
