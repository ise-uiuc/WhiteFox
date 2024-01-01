
class Model(torch.nn.Module):
    def __init__(self, d_model=200, n_heads=4, dropout_p=0.5):
        super().__init__()
        self.wq = torch.nn.Linear(d_model, d_model)
        self.wk = torch.nn.Linear(d_model, d_model)
        self.wv = torch.nn.Linear(d_model, d_model)
        self.scale_factor = math.sqrt(d_model)
        self.attn_dropout = torch.nn.Dropout(dropout_p)

    def forward(self, q, k, v, att_mask=None):
        wq, wk, wv = self.wq(q), self.wk(k), self.wv(v)
        att_score = torch.bmm(wq, wk.transpose(1, 2)) / self.scale_factor
        att_score += att_mask
        att_weight = self.attn_dropout(torch.nn.functional.softmax(att_score, dim=-1))
        output = torch.bmm(att_weight, wv)
        return output

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(100, 200)
k = torch.randn(100, 200)
v = torch.randn(100, 200)
att_mask = torch.zeros(100, 100).bool()
