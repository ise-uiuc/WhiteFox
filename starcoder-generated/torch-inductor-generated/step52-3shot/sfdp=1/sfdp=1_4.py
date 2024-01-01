
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention_dropout = nn.Dropout(config.attention_dropout)
        self.dropout = nn.Dropout(config.dropout)
        self.output_layer = nn.Linear(config.attention_dim, self.output_dim)
     
    def forward(self, query, key, value, mask):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(math.sqrt(query.size(-1)))

        if mask is not None:
            if not hasattr(mask, "dtype"):
                mask = mask.byte()
            if len(mask.size()) < len(scaled_qk.size()):
                mask = mask.unsqueeze(1)
            mask = mask.unsqueeze(1).repeat(1, query.size(1), 1, 1)
            scaled_qk.masked_fill_(mask, -1e4)
        softmax_qk = torch.nn.functional.softmax(scaled_qk, dim=-1)
        dropout_qk = self.attention_dropout(softmax_qk)
        output = torch.matmul(dropout_qk, value)
        return self.dropout(output)

# Initializing the model
model = Model()

