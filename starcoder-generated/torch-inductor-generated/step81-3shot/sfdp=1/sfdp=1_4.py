
class Model(torch.nn.Module):
    def __init__(self, dim_model=512, dim_feedforward=512, n_head=8, n_layer=6, dropout_p=0.5, dim_out=1000):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        for l in range(n_layer):
            layer = torch.nn.TransformerEncoderLayer(
                d_model=dim_model,
                nhead=n_head,
                dim_feedforward=dim_feedforward,
                dropout=dropout_p
            )
            norm = torch.nn.LayerNorm(dim_model)
            self.layers.append(layer)
            self.norms.append(norm)
        self.linear = torch.nn.Linear(1024, dim_out)
 
    def forward(self, x1):
        v1 = self.layers[0](x1)
        v2 = self.norms[0](v1)
        for l in range(1, len(self.layers)):
            v3 = self.layers[l](v2)
            v4 = self.norms[l](v3)
            v2 = v4 + v2
        v5 = v2.mean(dim=1)
        v6 = v5.mean(dim=1)
        v7 = torch.relu(v6)
        v8 = self.linear(v7)
        return v8

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(5, 64, 1000)
