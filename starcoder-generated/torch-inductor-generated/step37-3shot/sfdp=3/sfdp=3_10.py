
class Model(torch.nn.Module):
    def __init__(self, d=512, h=8, dropout=0.):
        super().__init__()
        self.d = d
        self.h = h
        self.dropout = dropout
        self.linear_layers = nn.ModuleList([nn.Linear(d, d) for _ in range(2)])
        self.norm_layers = nn.ModuleList([nn.LayerNorm(d) for _ in range(3)])
        
    def attention(self, x):
        y = self.norm_layers[0](x)
        y = self.linear_layers[0](y)
        y = self.linear_layers[1](y).softmax(-2)
        y = self.norm_layers[1](y)
        output = self.norm_layers[2](x)
        output = output * y
        output = output.sum(-2)
        return output
    
    def forward(self, x):
        return self.attention(x)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
