
class Model(torch.nn.Module):
    def __init__(self, embedding_size=256, heads=8, dropout_p=0.3, seq_len=512, vocab_len=10000, hidden_size=2048):
        super().__init__()
        
        self.token_embed = torch.nn.Embedding(vocab_len + 1, embedding_size)
        self.transformer = torch.nn.TransformerEncoderLayer(
            embedding_size, heads, hidden_size, dropout_p, norm_first=True)
        self.lm_head = torch.nn.Linear(embedding_size, vocab_len + 1)
    
        self.positional_embedding = torch.nn.Parameter(
            torch.randn(seq_len + 1, embedding_size))
        self._init_weights()
        
    # Note: this method is just used to initialize positional embedding and token embeddings.
    def _init_weights(self):
        initrange = 0.1
        self.token_embed.weight.data.uniform_(-initrange, initrange)
        self.positional_embedding.data.uniform_(-initrange, initrange)
    
    def forward(self, x1):
        embed = self.positional_embedding + self.token_embed(x1)
        
        out = self.transformer(embed)
        
        return self.lm_head(out)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 512)
