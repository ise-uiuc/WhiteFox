
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.key = torch.nn.Parameter(torch.randn(128))
    def forward(self, x1):
        q = x1[:,0:8,:,:]
        k = x1[:,8:16,:,:]
        v = x1[:,16:,:,:]
        w = torch.cat([q,k,v], dim=1)
        inv_scale = math.sqrt(w.size(1))
        scaled_dot_product = torch.matmul(w, w.transpose(-2, -1)) / inv_scale
        attention_weights = scaled_dot_product.softmax(dim=-1)
        output = attention_weights.matmul(w)
        return output.split(v.size())
# Inputs to the model
x1 = torch.randn(1, 24, 64, 64)
