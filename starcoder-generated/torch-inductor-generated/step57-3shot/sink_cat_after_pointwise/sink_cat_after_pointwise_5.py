
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = x.to(dtype=torch.float)
        y = torch.split(x, dim=0, split_size_or_sections=2)
        for t_i in y:
            s = torch.split(t_i, dim=1, split_size_or_sections=2)
#            t = [a + b for (a, b) in zip(s, tuple(t_i.clone().chunk(2, 1)))]
            t = torch.cat((s[0]+s[1],s[1]+s[0]), dim=0)
        return t
# Inputs to the model
x = torch.arange(1, 26).view(5, 5)
x = x.type(torch.LongTensor).to(torch.float32)
