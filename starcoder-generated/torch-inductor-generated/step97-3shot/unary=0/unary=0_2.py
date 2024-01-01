
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv12 = torch.nn.Conv2d(11, 21, 1, stride=1, padding=0)
    def forward(self, x10, x21):
        v15 = torch.tensor([1.0000001800231934, 0.9970238962173462])
        v38 = x10 * v15
        v16 = x21 * v15
        v5 = v16[0, 0, 1, 1]
        v39 = v38 + v5
        v40 = v38 * v39
        v17 = self.conv12(v40)
        v18 = v17 * 0.5
        v19 = v17 * v17
        v20 = v19 * v17
        v21 = v20 * 0.044715
        v22 = v17 + v21
        v23 = v22 * 0.7978845608028654
        v24 = torch.tanh(v23)
        v25 = v24 + 1
        v26 = v18 * v25
        return v26
# Inputs to the model
x10 = torch.randn(1, 11, 13, 11)
x21 = torch.randn(1, 11, 13, 11)
