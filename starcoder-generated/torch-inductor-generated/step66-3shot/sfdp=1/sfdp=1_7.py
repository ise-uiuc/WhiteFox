
class Model(torch.nn.Module):
    def __init__(self, dim_m, dim_i, dim_o):
        super().__init__()
        self.dim_m = dim_m
        self.dim_i = dim_i
        self.dim_o = dim_o
        self.scale_factor = 1.0 / math.sqrt(dim_m)
        self.linear_q = torch.nn.Linear(dim_m, dim_o)
        self.linear_k = torch.nn.Linear(dim_m, dim_o)
        self.linear_v = torch.nn.Linear(dim_m, dim_o)
 
    def forward(self, x1, x2, x3):
        q, k, v = self.linear_q(x1), self.linear_k(x2), self.linear_v(x3)
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(self.scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model(256, 28, 10)

# Inputs to the model, the shape can vary
x1 = torch.randn(17, 256)
x2 = torch.randn(17, 256)
x3 = torch.randn(17, 256)
