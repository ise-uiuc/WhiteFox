
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q = torch.nn.Linear(in_features, out_features)
        self.k = torch.nn.Linear(in_features, out_features)
        self.v = torch.nn.Linear(in_features, out_features)
 
    def forward(self, q, k, v):
        q = self.q(q)
        k = self.k(k)
        v = self.v(v)
        qk = torch.matmul(qk, k.transpose(-2, -1))
        scaled_qk = qk.div(1/sqrt(out_features))
        softmax_qk = scaled_qk.softmax(dim=-1)
        output = softmax_qk.matmul(v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(batch_size, in_features)
x2 = torch.randn(batch_size, in_features)
x3 = torch.randn(batch_size, in_features)
