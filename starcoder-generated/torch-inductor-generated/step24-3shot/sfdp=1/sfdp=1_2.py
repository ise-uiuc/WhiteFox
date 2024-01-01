
class Model(torch.nn.Module):
    def __init__(self, size, dropout=0.0):
        super().__init__()
        self.size = size
        self.dropout = dropout
        self.dropout_module = torch.nn.Dropout(dropout)
 
    def forward(self, x1, x2):
        v1 = torch.matmul(x1, x2.transpose(-2, -1))
        v2 = v1.div(float(self.size) ** -0.5)
        v3 = torch.nn.functional.softmax(v2, dim=-1)
        v4 = self.dropout_module(v3)
        v5 = torch.matmul(v3, x2)
        return v4, v5

# Initializing the model
m = Model(size=64)

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
x1 = torch.randn(1, 3, 64, 64, requires_grad=False)
x2 = torch.randn(1, 3, 64, 64, requires_grad=False)
__0_output__, __1_output__ = m(x1, x2)
_0_output_ = __0_output__.detach()
_1_output_ = __1_output__.detach()
torch.autograd.Variable(_0_output_) + torch.autograd.Variable(_1_output_)

dropout_p = 0.5
# Model 1
m = Model(size=64)
# Initializing the model
m_dropout = torch.nn.Dropout(dropout_p)
m_dropout.eval()
m_dropout.parameters()

# Model 2
weight = m_dropout.weight

x = torch.randn(20, 16)
