
class Model(torch.nn.Module):
    def __init__(self, d_model=512, n_head=8, dropout=0.1, depth=6):
        super().__init__()
        layer = []
        for _ in range(depth):
            layer.append(MultiheadAttention(n_head, d_model))
            layer.append(nn.Dropout(p=dropout))
        self.layers = nn.ModuleList(layer)
        self.out = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, 10))
 
    def forward(self, x, y):
        _y = x
        for layer in self.layers:
            _y = layer(_y, _y, _y)
        return self.out(_y)

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 10, 512)
y = torch.randn(1, 10, 512)

# Outputs from the model
