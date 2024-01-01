
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        t1 = torch.Tensor.from_numpy(np.array([3.346673494962252e-05, -1.715499986568743e-05, -1.633929946167151e-05, 3.589893498520263e-05, -2.597440712399695e-05, 6.452198813599724e-05])).cuda().float()
        t2 = t1.to(torch.double) # convert from float32 to float64
        return t2
# Inputs to the model
input = torch.randn(6)
