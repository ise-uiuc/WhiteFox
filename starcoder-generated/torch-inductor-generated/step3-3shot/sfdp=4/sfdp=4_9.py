
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, attn_mask):
        attn_mask = torch.zeros(attn_mask.size()).masked_fill_(attn_mask, -float("inf"))
        # print("attn_mask:", attn_mask.shape)
        attn_mask = torch.transpose(attn_mask, 0, 1)
        # print("attn_mask:", attn_mask.shape)
        v = input @ input.transpose(1,2)
        mask = (1.0-attn_mask) * -10000.0
        v = v + mask
        w = torch.empty_like(v).uniform_(-0.1,0.1)
        w = torch.softmax(w, dim=-1)
        for i in range(w.shape[-1]):
            w[:,i] = w[:,i] / torch.norm(w[:,i], 1)

        return (w @ input)
# Initializing the model
m = Model()

# Inputs to the model
input = torch.randn(20,30,10)
attn_mask = torch.zeros(20,10)
for i in range(20):
    for j in range(10):
        if i == j:
            attn_mask[i][j] = 1

