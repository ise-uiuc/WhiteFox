
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(128, 64, padding_idx=1)
        # Positional embeddings
        self.pos = nn.Parameter(torch.randint(-128, 128, (4, 10)))
        self.dropout = torch.nn.Dropout(0.1)
    
 
    def forward(self, x1):
        emb = self.emb(x1)
        pos = self.pos[:, :x1.shape[1]]
        emb = emb + pos
        emb = self.dropout(emb)
        return emb


# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randint(0, 3, (1, 4))
