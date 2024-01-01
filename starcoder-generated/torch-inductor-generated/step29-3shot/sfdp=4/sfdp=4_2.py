
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.TransformerEncoder(
            encoder_layer=torch.nn.TransformerEncoderLayer(
                d_model=64,
                nhead=4
            ),
            num_layers=2,
            norm=torch.nn.LayerNorm(64)
        )
    def forward(self, src, src_mask):
        output = self.layer(src, src_mask=src_mask)
        return output
# Inputs to the model
src = torch.randn(10, 32, 512)
src_mask = (torch.rand(32, 32) > 0).float().cuda()
