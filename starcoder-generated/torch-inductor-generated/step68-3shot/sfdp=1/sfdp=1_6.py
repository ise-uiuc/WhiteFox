
class Model(torch.nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.query = torch.nn.Linear(256, 256, bias=True)
        self.key = torch.nn.Linear(256, 256, bias=True)
        self.value = torch.nn.Linear(256, 256, bias=True)
 
    def forward(self, in1, in2, in3):
        q = self.query(in1)
        k = self.key(in2)
        v = self.value(in3)
        q *= (1. / math.sqrt(self.query.out_features))
        k *= (1. / math.sqrt(self.key.out_features))
        return torch.matmul(q, k.transpose(-2, -1))

# Initializing the model
m = Model(2)

# Inputs to the model
in1 = torch.randn(1, 40, 256)
in2 = torch.randn(1, 30, 256)
in3 = torch.randn(1, 30, 256)
