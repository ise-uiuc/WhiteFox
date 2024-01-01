
class TransformerEncoderLayer(nn.TransformerEncoderLayer):
    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        return super().forward(src, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
m = TransformerEncoderLayer(d_model=2, nhead=1) # d_model and nhead should be different from the model defined in a previous question

# Initializing the model
x1 = torch.rand(2, 3, 2)
