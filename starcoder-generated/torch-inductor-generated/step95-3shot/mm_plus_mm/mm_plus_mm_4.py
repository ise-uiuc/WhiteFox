
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, inputs1, inputs2, inputs3, inputs4):
        layer_weights = {}
        layer_weights["t1"] = torch.mm(inputs1, inputs2)
        layer_weights["t2"] = torch.mm(inputs3, inputs4)
        layer_weights["t3"] = layer_weights["t1"] + layer_weights["t2"]
        return layer_weights["t3"]
# Inputs to the model
inputs1 = torch.randn(166, 320)
inputs2 = torch.randn(320, 1024)
inputs3 = torch.randn(166, 320)
inputs4 = torch.randn(320, 1024)
