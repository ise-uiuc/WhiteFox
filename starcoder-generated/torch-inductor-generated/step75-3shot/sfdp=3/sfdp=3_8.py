
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.key = torch.nn.Parameter(torch.empty(1, 40, 1024))
        self.dropout = torch.nn.Dropout(dropout_p)
        self.dropout.train()

    def forward(self, x1, x2, x3):
        qk = torch.matmul(x1, self.key.transpose(-2, -1))
        scale_factor = torch.sqrt(qk.size(-1))
        scaled_qk = qk * scale_factor
        softmax_qk = softmax(scaled_qk, dim=-1)
        dropout_qk = self.dropout(softmax_qk)
        output = dropout_qk.matmul(x2)
        return output

# Initializing parameters of the model
m = Model()

# Initializing model parameters with Xavier initialization
for n, p in param_init(m):
    p.data.fill_(0.1)
    if p.requires_grad:
        n.register_hook(lambda param, grad: param * 0.1)

# Inputs to the model
x1 = torch.randn(batch size, T, 40)
x2 = torch.randn(batch size, T, 1024)
x3 = torch.randn(batch size, 1024)

# Generating outputs
m.train()
