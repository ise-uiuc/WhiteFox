
class Model(torch.nn.Module):
    def __init__(self, nhead=8, embed_dim=48, num_encoder_layers=12, num_decoder_layers=12, intermediate_size=1080):
        super().__init__()
        self.encoder = torch.nn.TransformerEncoder(torch.nn.TransformerEncoderLayer(embed_dim, nhead, dim_feedforward=intermediate_size), num_encoder_layers)
        self.decoder = torch.nn.TransformerDecoder(torch.nn.TransformerDecoderLayer(embed_dim, nhead, dim_feedforward=intermediate_size), num_decoder_layers)
        self.linear = torch.nn.linear(embed_dim, 8)
        self.encoder_embed = torch.nn.Embedding(256, embed_dim)
        self.decoder_embed = torch.nn.Embedding(256, embed_dim)

    def forward(self, input, target):
        x = self.encoder_embed(self, input)
        y = self.decoder_embed(self, target)
        out = self.decoder(self, y, x)
        return self.linear(self, out)

# Initializing the model
m = Model()

# Inputs to the model
input = torch.randint(0, 256, (10, 20))
# The following line randomly generates the target sequence. This line is for demo purpose only
target = torch.randint(0, 256, (10, 15))
