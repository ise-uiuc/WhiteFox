
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.matmul1 = torch.nn.Linear(128, 8)
        self.matmul2 = torch.nn.Linear(128, 8)
    def forward(self, inp1, inp2, inp3):
        # Apply self-attention
        qk1 = self.matmul1(inp1).reshape(56, 4, 32)
        qk2 = self.matmul2(inp2).reshape(56, 32, 4)
        qk = torch.matmul(qk1, qk2.transpose(-2, -1))
        scale_factor = 1. / np.sqrt(np.shape(inp1)[-1])
        scaled_qk = scale_factor * qk
        attn_weights = torch.softmax(scaled_qk, dim=-1)
        dropout_attn_weights = torch.nn.functional.dropout(attn_weights, p=0.5)
        out = torch.matmul(dropout_attn_weights, inp3)
        return out

# Initializing the model
m = Model()

# Inputs to the model
inp1 = torch.randn(56, 128)
inp2 = torch.randn(56, 4, 32)
inp3 = torch.randn(56, 32, 4)
