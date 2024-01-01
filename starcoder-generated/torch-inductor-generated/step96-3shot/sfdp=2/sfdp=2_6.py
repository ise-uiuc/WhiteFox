
class Model(torch.nn.Module):
    def __init__(self, embedding_dim, num_heads, num_words, dim_per_head):
        super().__init__()
        self.embedding = torch.nn.Embedding(embedding_dim, num_words, 0)
        self.encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(embed_dim=embedding_dim, num_heads=num_heads, dim_feedforward=1024), num_layers=2)
        self.decoder = torch.nn.TransformerDecoder(
            torch.nn.TransformerDecoderLayer(embed_dim=embedding_dim, num_heads=num_heads, dim_feedforward=1024), num_layers=2)
 
    def forward(self, x1, x2, x3):
        x = self.embedding(x1) + self.embedding(x2)
        x = self.encoder(x, x3)
        y = torch.ones_like(x)
        x = self.decoder(y, x)
        return x

# Initializing the model
m = Model(num_words=512, embedding_dim=512, num_heads=8, dim_per_head=128)

# Inputs to the encoder
x1 = torch.randint(512, [1, 20])
x2 = torch.randint(512, [2, 20])
x3 = torch.randn(2, 20, 512)

# Outputs from the decoder
y = torch.ones_like(x3)
