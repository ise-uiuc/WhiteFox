
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query = torch.nn.Linear(4, 6)
        self.key = torch.nn.Linear(5, 8)
        self.value = torch.nn.Sequential(
            torch.nn.Linear(5, 7),
            torch.nn.Tanh(),
        )
 
    def forward(self, q, k, v, mask):
        qk = self.query(q) @ self.key(k).transpose(-2, -1)
        attn_mask = mask.unsqueeze(1)
        qk = qk + attn_mask
        attn_weights = torch.softmax(qk, dim=-1)
        output = attn_weights @ self.value(v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(1, 3, 4) # Shape: [1, 3, 4]
k = torch.randn(4, 1, 5) # Shape: [4, 1, 5]
v = torch.randn(4, 1, 5) # Shape: [4, 1, 5]
mask = torch.tensor([[False, False, True], [False, False, False], [True, True, True]], dtype=torch.bool) # Shape: [3, 3]
