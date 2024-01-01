
class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads=8, dropout=0.0) -> None:
        
        super(MultiHeadAttention, self).__init__()
        self.n_heads = num_heads
        self.d_model = d_model
        self.scale_factor = 1 / np.sqrt(self.d_model)

        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)
    
    def forward(self, query, key, value:Optional[torch.Tensor]=None, mask:Optional[torch.Tensor]=None):
        