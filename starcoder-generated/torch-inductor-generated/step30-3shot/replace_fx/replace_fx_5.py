
class testModel(torch.nn.Module):
    def forward(x):
        o1 = torch.nn.functional.dropout(x, p=0.8)
        o2 = torch.rand_like(x)
        return o1
# Inputs to the model
x = torch.randn([])
