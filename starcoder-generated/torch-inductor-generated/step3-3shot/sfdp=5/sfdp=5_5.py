
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.trans_layer_norm = torch.nn.LayerNorm(768, eps=1e-6)
        self.self_attention = torch.nn.MultiheadAttention(768, 4, dropout=0.2, bias=True)
        self.trans_fc = torch.nn.Linear(768, 3072)
 
    def forward(self, x2):
        v2 = self.query # Initialize a query tensor of shape [-1, 4, 1, 768]. The value will be replaced if the input format differs
        