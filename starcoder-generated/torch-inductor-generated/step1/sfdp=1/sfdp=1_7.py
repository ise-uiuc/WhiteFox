
class Model(torch.nn.Module):
    def __init__(self, qk_size, v_size, dropout_p=0.5):
        super().__init__()
        self.proj = torch.nn.Linear(qk_size, v_size)

    def forward(self, query, key, value, inv_scale_factor):
        out = self.proj(torch.nn.functional.softmax((torch.matmul(query, key.transpose(-2, -1)) / inv_scale_factor), dim=-1))
        return torch.nn.functional.dropout(out, p=dropout_p, train=self.training)

# Initializing the model
m = Model(qk_size=3, v_size=4)

# Inputs to the model
query = torch.randn(2, 5, 3)
key = torch.randn(2, 3, 8)
value = torch.randn(2, 3, 4)
inv_scale_factor = math.sqrt(3)
