
class Model(torch.nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, num_layers, dropout_p):
        super().__init__()
        self.encoder_layers = torch.nn.ModuleList([TransformerEncoderLayer(d_model,
                                                                          nhead,
                                                                          dim_feedforward,
                                                                          dropout_p) for i in range(num_layers)])
 
    def forward(self, x1, x2):
        for layer in self.encoder_layer:
            x2 = layer(x1, x2)
        return x2

# Initializing the model
m = Model(d_model=1,
          nhead=1,
          dim_feedforward=1,
          num_layers=1,
          dropout_p=0.1)

# Inputs to the model
x1 = torch.randn(1, 1, 1)
x2 = torch.randn(1, 1, 1)
