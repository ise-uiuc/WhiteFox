
import math
class TransformerModel(nn.Module):
    def __init__(self, ntoken: int, ninp: int, nhead: int):
        super().__init__()
        self.ninp = ninp
        self.model_dim=ninp
        self.wih = nn.Linear(ninp, 5 * ninp)
        self.whh = nn.Linear(ninp, 5 * ninp)
        self.bhi = nn.Linear(ninp, 5 * ninp)
        self.bhh = nn.Linear(ninp, 5 * ninp)
        self.decoder = nn.Linear(ninp, ntoken)
 
        self.encoder = nn.Embedding(ntoken, ninp)
        self.model_dim=ninp
        self.positional_encoder = PositionalEncoder(ninp)
 
    def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
        # Embed source.
        src = self.encoder(src) * math.sqrt(self.model_dim)
        src = self.positional_encoder(src)
        h = self.wih(src)
        h = h.view(-1, src.size(0), 5, self.model_dim)
        h = h.transpose(0, 1)
        q, k, v = h[:, :, 0:1, :], h[:, :, 1:2, :], h[:, :, 2:3, :]
        h = h[:, :, 3:5, :]
 
        # Attention.
        k = k.transpose(-2, -1)
        q = q.transpose(-2, -1)
        src2 = (q @ k) / math.sqrt(self.model_dim)
        src2 = src2 + src_mask
        src2 = torch.softmax(src2, dim=-1)
        output = (src2 @ v) + h
 
        # Transform to full dimension.
        output = output.transpose(0, 1)
        output = output.contiguous()
        output = output.view(-1, model_dim)
 
        # Output projection.
        output = self.decoder(output)
        return output
 
class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pe.require_grad = False
 
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
 
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
 
    def forward(self, x):
        return self.pe[:, :(x.size(1)), :]

# Initializing the model
model = TransformerModel(ntoken=32, ninp=32, nhead=2)

# Input to the model


src = torch.randint(0, 32, (20, 5), dtype=torch.long)
src_mask = 0.2
