
class Model(torch.nn.Module):
    def __init__(self, input, hidden):
        super().__init__()
        assert input % hidden == 0
        self.hidden_size = hidden
        self.output_size = input // hidden

    def forward(self, x):
        v = x.view(-1, self.hidden_size, self.output_size)
        q = v.permute(0, 2, 1).contiguous().view(-1, self.output_size, self.hidden_size)
        k = v.contiguous().view(-1, self.output_size, self.hidden_size)
        qk = torch.nn.functional.softmax((q * k).sum(-1), -1)
        output = torch.einsum("bxy,bxz->byz", v, qk.unsqueeze(-1)).view(x.size())
        return output

# Initializing the model
m = Model(512, 64)

# Inputs to the model
x1 = torch.randn(32, 512, 8, 8)
