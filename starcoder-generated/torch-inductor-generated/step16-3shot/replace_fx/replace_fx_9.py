
class model(torch.nn.Module):
    def forward(self, x):
        return x.view(-1, )[0]
# Inputs to the model
x1 = torch.randn([1, 2])
