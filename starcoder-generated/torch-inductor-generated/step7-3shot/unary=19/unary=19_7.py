
class Model(torch.nn.Module):
    def __init__(self, d_model, nhead, d_hid, nlayers):
        super(Encoder, self).__init__()

        self.nhead = nhead
        self.d_model = d_model
        self.d_hid = d_hid

        self.input_linear = nn.Linear(d_input, d_model)

        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)

    def forward(self, src, src_mask):
        # src = [batch size, src len, d_model]
        # src_mask = [batch size, src len]

        src = self.input_linear(src) * math.sqrt(self.d_model)
        src = F.relu(src)
        src = self.transformer_encoder(src, src_mask) * math.sqrt(self.d_model)

        # src = [batch size, src len, d_model]

        return src

# Initializing the model
m = Encoder(src_vocab, d_model, nheads, dim_feedforward, nlayers, dropout)

# Inputs to the model
x1 = torch.randn(16, 10, 512)
mask1 = torch.randn(16, 1, 1, 10)
