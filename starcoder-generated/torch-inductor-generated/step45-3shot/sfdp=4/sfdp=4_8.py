
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, Q5, k, v, mask):
        qk = Q5 @ k.transpose(-2, -1) / math.sqrt(Q5.size(-1))
        qk = qk + mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ v

        output2 = self.layer1(output)
        output3 = output2 + output 
        x = self.layer2(output3)
        output4 = self.layer3_b(x)
        output5 = output4 + output3
        output6 = self.layer3_a(output5)
        ouput7 = output6 + output
        attention_map = torch.softmax(qk.mean(0), 0)
        attention_map = attention_map / torch.sum(attention_map, 0, keepdim=True)
        return output
# Inputs to the model
Q = torch.randn(1, 64, 56, 56)
K = torch.randn(1, 64, 56, 56)
V = torch.randn(1, 64, 56, 56)
mask = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
