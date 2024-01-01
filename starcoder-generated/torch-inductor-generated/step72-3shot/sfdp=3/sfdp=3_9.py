
class Model(torch.nn.Module):
    def forward(self, x1, x2):
        q = x1
        k = x2
        q_ = q.reshape(1, q.shape[0], q.shape[1], 1)
        k_ = k.reshape(k.shape[0], 1, k.shape[1], k.shape[2])
        m = q_ * k_
        t = torch.matmul(q, k.transpose(2, 3))
        s = t * scale_factor
        f = s.softmax(-1) + 1e-5
        o = torch.matmul(f, v)
        d = torch.nn.functional.dropout(o, p=dropout_p)
        return d

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(10, 256)
x2 = torch.randn(10, 256, 16, 16)
