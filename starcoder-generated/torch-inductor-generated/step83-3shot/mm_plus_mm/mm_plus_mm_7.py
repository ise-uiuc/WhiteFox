
class Model(torch.nn.Module):
    def forward(self, inputs):
        t1 = torch.mm(inputs[0], inputs[1]) # Matrix multiplication between 0 and 1
        t2 = torch.mm(inputs[2], inputs[3]) # Matrix multiplication between 2 and 3
        t3 = torch.mm(inputs[4], inputs[5]) # Matrix multiplication between 4 and 5
        return t1 + t2 + t3 # Addition of the results
# Inputs to the model
inputs = [torch.randn(8, 8), torch.randn(8, 8), torch.randn(8, 8), torch.randn(8, 8), torch.randn(8, 8), torch.randn(8, 8)]
