
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.key = Parameter(torch.Tensor(8, 8, 16, 16))
        self.scale_factor = 10.0
        self.dropout_p = 0.1
        self.value = torch.nn.Parameter(torch.Tensor(8, 8, 16, 16))
 
    def forward(self, x1):
        k = self.key
        v = self.value
        s = self.scale_factor
        p = self.dropout_p
        _1 = torch.matmul(x1, k.transpose(-2, -1))
        _2 = _1 * s
        _3 = _2.softmax(dim=-1)
        _4 = torch.nn.functional.dropout(_3, p=p)
        _5 = _4.matmul(v)
        return _5

# Initializing the model
s = 4
seed = 0
torch.manual_seed(seed)
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8, s, s)
