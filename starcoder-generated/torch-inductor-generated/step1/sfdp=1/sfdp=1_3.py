
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.n = 8
        self.c = 4096
        self.dropout_p = 0.5
        self.inv_scale_factor = 1.0 / math.sqrt(self.c)
        self.dropout = torch.nn.Dropout(p=self.dropout_p)
 
    def forward(self, x):
        k = torch.randn((self.n, self.n))
        v = torch.randn((self.n, self.c))
        q = torch.randn((self.n, self.c))
        z = self.dropout(torch.matmul(q, k.transpose(-2, -1)).div(self.inv_scale_factor)).softmax(dim=-1).matmul(v)
        return z

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, model.n, model.c)
