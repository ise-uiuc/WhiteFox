
class Model(torch.nn.Module):
    def __init__(self, n_layer, n_head, n_embd):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        encoder_layer = Transformer.EncoderLayer(n_embd)
        self.transformer = Transformer.Encoder(encoder_layer, n_layer)
        self.n_embd = n_embd
 
        self.decoder = nn.Linear(768, 10)
 
    def forward(self, x1):
        v1 = self.transformer(x1, None)
        v2 = v1.view(-1, self.n_embd)
        return self.decoder(v2)

# Initializing the model
m = Model(3, 1, 1)

# Inputs to the model
x2 = torch.randn(10, 3, 64, 64)
