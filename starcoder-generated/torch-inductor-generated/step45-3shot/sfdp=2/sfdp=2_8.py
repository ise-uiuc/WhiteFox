
class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model, nheads, dropout=0.1):
        super().__init__()
        self.encoder_layer = torch.nn.TransformerEncoderLayer(d_model, nheads, dropout)
 
    def forward(self, x1):
        v1 = x1.permute(1, 0, 2)
        v2 = self.encoder_layer(v1)
        result = v2.permute(1, 0, 2)
        return result

# Inputs to the model
x1 = torch.randn(32, 256, 512)
