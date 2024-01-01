
class TransformerModel(nn.Module):
    def __init__(self, ninp, nhead, nhid, nlayers, dropout=0.5):
        super().__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer

        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout) # Create an encoder layer
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers) # Create an encoder
        
        self.src_mask = None
 
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
 
    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0)!= len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask
 
        return self.transformer_encoder(src, self.src_mask)

# Initializing the model
ninp, nhead, nhid, nlayers = 100, 10, 10, 10
m = TransformerModel(ninp, nhead, nhid, nlayers)

# Inputs to the model
x1 = torch.randn(64, 32, 100)
