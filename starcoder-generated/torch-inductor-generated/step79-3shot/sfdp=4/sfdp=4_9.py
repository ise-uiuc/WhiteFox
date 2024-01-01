
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    def forward(self, inputs, inputs2, inputs3):
        i = torch.randn(1, 64, 56, 56)
        i2 = torch.randn(1, 64, 56, 56)
        i3 = torch.randn(1, 64, 56, 56)
        o = i @ i2.transpose(-2, -1) / math.sqrt(i.size(-1))
        o = o + attention_mask
        output = o @ i3
        return output
# Inputs to the model
input1 = torch.randn([1, 64, 56, 56])
input2 = torch.randn([1, 64, 56, 56])
input3 = torch.randn([1, 64, 56, 56])
attention_mask = (torch.rand(1, 56, 56) > 0.7).float().masked_fill_(
attention_mask.bool(),
-1000000000.0
)
