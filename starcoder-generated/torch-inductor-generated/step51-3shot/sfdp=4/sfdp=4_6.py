
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input1, input2, input3, input4):
        qk = input1 @ input2.transpose(-2, -1) / math.sqrt(input1.size(-1))
        qk = qk + input4
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ input3
        return output

