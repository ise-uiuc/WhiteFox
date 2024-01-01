
class Model(torch.nn.Module):
    def __init__(self, n_heads, d_qkv, d_model):
        super().__init__()
 
        self.n_heads = n_heads
        self.d_qkv = d_qkv
        self.d_model = d_model
 
        self.W_q = torch.nn.Linear(d_model, d_qkv)
        self.W_k = torch.nn.Linear(d_model, d_qkv)
        self.W_v = torch.nn.Linear(d_model, d_qkv)
        self.W_out = torch.nn.Linear(d_qkv, d_model)
 
    def _split_heads(self, x, is_key=False):
        