
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_norm = torch.nn.LayerNorm(d_hid, eps=1e-30)
        self.w_query = torch.nn.Linear(d_hid, d_hid)
        self.w_key = torch.nn.Linear(d_hid, d_hid)
        self.w_value = torch.nn.Linear(d_hid, d_hid)
 
    def forward(self, x1, x2, x3, x5):
        v1 = self.layer_norm(x4)
        v2 = self.w_query(v1)
        v3 = self.w_key(x1)
        v4 = self.w_value(x2)
        v5 = v2 @ v3.transpose(-2, -1) / math.sqrt(d_hid)
        v6 = v5 + x2
        v7 = torch.softmax(v6, -1)
        x6 = v7 @ v4
        return x6

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(4, 10, d_hid)
x2 = torch.randn(4, 10, d_hid)
x3 = torch.randn(4, d_hid)
x5 = torch.randn(4, 1, d_hid)
