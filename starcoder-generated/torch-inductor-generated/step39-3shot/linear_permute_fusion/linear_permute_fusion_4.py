
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block = torch.nn.TransformerEncoderLayer(2, 2)
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        self.block.self_attn.softmax_temp = 1
        v1 = x1.permute(1, 0, 2)
        v4 = v1 * 1
        v2 = self.block(v1, src_key_padding_mask=(v4 == 0))
        v3 = v2 * 1
        return v3
# Inputs to the model
x1 = torch.randn(2, 3, 2)
