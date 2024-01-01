
class Model(torch.nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers=1):
        super().__init__()
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_encoder_layers)
 
    def forward(self, src, src_mask=None):
        output = self.transformer_encoder(src, src_mask)
        return output

# Initializing the model
m = Model(d_model=128, nhead=4, num_encoder_layers=1)

# Inputs to the model
src = torch.randn(10, 64, 128)
src_mask = torch.randn(10, 1, 64, 64).round()
