
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj_0 = torch.nn.Linear(3, 16)
        self.proj_1 = torch.nn.Linear(16, 16)
        self.proj_2 = torch.nn.Linear(16, 8)
        self.dropout = torch.nn.Dropout(0.2)
 
    def forward(self, q, k, v):
        q = self.proj_0(q)
        k = self.proj_1(k)
        v = self.proj_2(v)
        qk = torch.matmul(q, k.transpose(-2, -1))
        scale_factor = qk.size(-1) ** -0.75
        scaled_qk = qk * scale_factor
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = self.dropout(softmax_qk)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(2, 3)
k = torch.randn(2, 3)
v = torch.randn(2, 8)
