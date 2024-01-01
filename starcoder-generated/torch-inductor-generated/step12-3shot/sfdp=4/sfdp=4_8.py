
class TransformerNet(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.embeddings = nn.Embedding(hparams.n_vocab, hparams.d_model)
        self.pos_embeddings = nn.Embedding(hparams.n_positions, hparams.d_model)
        self.drop = nn.Dropout(hparams.embd_pdrop)
        self.transformer = nn.Transformer(hparams.n_layers, hparams.n_heads,
                                         hparams.d_model, hparams.d_ff,
                                         hparams.dropout)
        self.ln1 = nn.LayerNorm(hparams.d_model)
        self.ln2 = nn.LayerNorm(hparams.d_model)
        self.linear = nn.Linear(hparams.d_model, hparams.n_vocab)

        self.register_buffer('position_ids', torch.arange(hparams.n_positions).expand((1, -1)))

    def forward(self, inp):
        embedded = self.embeddings(inp)
        embedded = self.drop(embedded)
        embedded = spatial_transformer_grid(embedded, self.position_ids)
        embedded = self.transformer(embedded)
        embedded = self.ln1(embedded)
        output = self.linear(embedded)
        return output
    
    def describe(self):
        print("The model contains the following components:")
        print("Embedding")
        print("Transformer")
        print("LayerNorm")
        print("Linear")

# Initializing the model
hparams = Hparams()
m = TransformerNet(hparams)

# Inputs to the model
x1 = torch.tensor([[3362, 1435]])
