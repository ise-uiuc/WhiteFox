
class Model(nn.Module):
    def forward(self, model):
        t1 = torch.mm(model, model)
        t2 = torch.mm(model, model)
        return t1 + t2
# Inputs to the model
model = torch.randn(10, 10)
