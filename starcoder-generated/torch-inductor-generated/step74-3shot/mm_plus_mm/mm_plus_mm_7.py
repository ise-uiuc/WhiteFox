
class Model(torch.nn.Module):
    def forward(self, inputs):
        out = torch.mm(inputs, inputs)
        out = torch.mm(inputs, inputs)
        outputs = torch.mm(out, inputs)
        return outputs
# Inputs to the model
inputs = torch.randn(5, 5)
