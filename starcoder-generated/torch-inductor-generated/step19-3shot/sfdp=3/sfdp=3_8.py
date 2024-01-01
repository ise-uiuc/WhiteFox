
class Model(torch.nn.Module):
    # Here we compute the dot product.
    # You should define a dot product operator in PyTorch, or use an existing one.
    def scaled_dot_product(self, query, key, scale_factor=1./math.sqrt(64)):
        return query.matmul(key.transpose(-2, -1)) * scale_factor

    # Here we can add one more layer to the model.
    def attention(self, q, k, v, dp=0.5):
        qk = self.scaled_dot_product(q, k)
        v = v.squeeze(1)
        qk_d = torch.nn.functional.dropout(
            torch.softmax(qk, dim=-1), dp)
        return qk_d.matmul(v).unsqueeze(1)

    def forward(self, q, k, v, dp=0.5):
        t1 = self.attention(q, k, v, dp)
        q = q.squeeze(1)
        t2 = self.attention(q, k, v, dp)
        t3 = t1 * t2
        return t3

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(1, 1, 64, 64)
k = torch.randn(1, 1, 64, 64)
v = torch.randn(1, 1, 64, 64)
