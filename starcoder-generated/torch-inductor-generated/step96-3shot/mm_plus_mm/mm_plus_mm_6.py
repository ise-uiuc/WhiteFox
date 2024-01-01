
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb_dim = emb_dim = 768
    def forward(self, input, attention_mask, training=True):
        c1 = torch.matmul(input, torch.transpose(input, 0, 2))
        c2 = torch.matmul(torch.transpose(attention_mask, 0, 1), attention_mask)
        c3 = torch.matmul(input, torch.transpose(attention_mask, 0, 1))
        c4 = torch.matmul(torch.transpose(attention_mask, 0, 1), input)
        out = c1 + c2 + c3 + c4
        return out
# Model inputs
input = torch.randn(4, 12)
attention_mask = torch.ones(4, 12)
torch.nn.Dropout(p=dropout_prob)
