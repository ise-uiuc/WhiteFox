
class Model(torch.nn.Module):
    def __init__(self, d_model=512, n_head=8, d_head=64, drop_ratio=0.1):
        super().__init__()
        self.scale_factor = torch.tensor(d_head ** -0.5)
        self.dropout_p = drop_ratio
        self.Q = torch.nn.Linear(d_model, d_head * n_head)
        self.K = torch.nn.Linear(d_model, d_head * n_head)
        self.V = torch.nn.Linear(d_model, d_head * n_head)
 
    def forward(self, x1):
        q = self.Q(x1)
        k = self.K(x1)
        v = self.V(x1)
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.mul(self.scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 512, 128, 10)
