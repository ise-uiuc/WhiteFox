
class Model(nn.Module):
    def __init__(self, dim, hidden_dim, num_layers, num_heads, mlp_dim, dropout_p=0.1):
        super().__init__()
        self.scale_factor = dim ** -0.5
        self.mlp_layers = nn.ModuleList()
        for _ in range(num_layers):
            mlp_layer = nn.Sequential(
                nn.Linear(dim, mlp_dim),
                nn.ReLU(inplace=True),
                nn.Linear(mlp_dim, dim),
                nn.Dropout(p=dropout_p),
            )
            self.mlp_layers.append(mlp_layer)
 
    def forward(self, x, attn_mask):
        input = x
        for mlayer in self.mlp_layers:
            x = input + mlayer(x)
        return x

# Initializing the model
m = Model(dim=64, hidden_dim=512, num_layers=6, num_heads=8, mlp_dim=2048)

# Inputs to the model
x = torch.randn(10, 64, 56, 56)
attn_mask = torch.ones(15, 56, 56, requires_grad=False)
