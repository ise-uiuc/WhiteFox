
class Model(nn.Module):
    def __init__(self, d_model, nhead, dropout=0):
        super().__init__()
        selff.d_model = d_model
        self.nhead = nhead
        self.dropout = nn.Dropout(p=dropout)

    def attention(self, query, key, value, mask=None):
        