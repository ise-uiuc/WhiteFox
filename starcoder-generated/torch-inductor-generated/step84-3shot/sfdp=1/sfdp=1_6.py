
class Model(torch.nn.Module):
    def __init__(self, num_tokens):
        super().__init__()
        self.wte = torch.nn.Embedding(num_tokens, d_model)
        self.wpe = torch.nn.Embedding(num_tokens, d_model)
        self.drop = torch.nn.Dropout(dropout_p)
 
    def forward(self, x, mask):
        x = self.drop(x)
        token_embeddings = self.wte(x)
        position_embeddings = self.wpe(torch.arange(x.size(1), device=device) + (mask.sum(1, keepdim=True) - 1))
        embeddings = token_embeddings + position_embeddings
        return embeddings * torch.sqrt(token_embeddings.size(-1))

# Initializing the model
m = Model(num_vocab)

# Generate dummy inputs to the model
x = torch.randint(low=0, high=256, size=(batch_size, 256))
mask = torch.ones((batch_size, 256), dtype=torch.bool, device=device)

# Inputs to the model
y = m(x, mask)

