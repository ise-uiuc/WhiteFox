
class Model(torch.nn.Module):
    def forward(self, x):
        v1 = torch.nn.functional.dropout(x, p=0.2, training=False)
        v2 = torch.rand_like(v1)
        r1 = v1 * v2
        r2 = r1.permute(0, 2, 1)
        r3 = torch.nn.functional.linear(r2, torch.tensor([[1.0, 0.0], [0.0, 1.0]]).T, v2.reshape(-1))
        return r1

# Initializing the model
m = Model()

