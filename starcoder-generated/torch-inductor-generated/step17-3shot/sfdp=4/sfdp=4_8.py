
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.n_head = 8
        self.dim_head = 32

    def forward(self, x1, x2, attn_mask=None):
        b1, s1, d1 = *x1.size(), x1.device
        b2, s2, d2 = *x2.size(), x2.device
        h = self.n_head
        dh = self.dim_head
        w1 = x1.reshape(b1, h, s1, dh)
        w2 = x2.reshape(b2, h, s2, dh)
        a = torch.softmax((w2@torch.swapaxes(w1, -2, -1))/math.sqrt(dh), -1)
        if attn_mask is not None:
            a = a + attn_mask
        v1, v2 = w1@a, w2@torch.swapaxes(a, -2, -1)
        v3 = torch.cat((v1, v2), -1).view(b1, s1, 2*h*dh)
        output = torch.relu(self.to_out(v3))
        return output
        
# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.rand(1, 20, 128)
x2 = torch.rand(1, 10, 128)

