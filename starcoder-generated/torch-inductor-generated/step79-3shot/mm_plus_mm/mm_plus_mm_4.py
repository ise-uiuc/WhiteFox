
class Model(torch.nn.Module):
    def forward(self, in0):
        out1 = torch.mm(torch.mm(in0, in0), in0)
        out2 = torch.mm(in0, in0)
        out3 = torch.mm(in0, torch.mm(in0, in0))
        return out2
# Inputs to the model
in0 = torch.randn(2, 2)
# model ends



# Print the model generated in this cell
print(Model())
