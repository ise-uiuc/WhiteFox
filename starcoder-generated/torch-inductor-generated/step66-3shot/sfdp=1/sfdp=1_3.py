
class Model(torch.nn.Module):
    def __init__(self, dropout_p=0.1, scale=100000):
        super().__init__()
        self.dropout_p = dropout_p
        self.scale = scale
 
    def forward(self, q, k, v):
        k_transpose = (k.transpose(-2, -1))
        v_transpose = (v.transpose(-2, -1))
        # 