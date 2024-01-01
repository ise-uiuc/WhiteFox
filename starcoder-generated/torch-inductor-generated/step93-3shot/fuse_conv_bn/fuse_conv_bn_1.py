
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, 3, 2)
        torch.manual_seed(1)
        self.conv2 = torch.nn.Conv2d(1, 16, 7, 3)
        self._register_state_dict_hook(self._test_hook)
        self.bn = torch.nn.BatchNorm2d(32, running_mean=torch.zeros([32]), affine=False)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn(x)
        return x
    def _test_hook(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        running_norm_state = {}
        for x in state_dict:
            if x.endswith('running_mean'):
                running_norm_state[x[16:]] = state_dict[x]
        for x in running_norm_state:
            state_dict['bn.running_mean'] += running_norm_state[x]
# Inputs to the model
x = torch.randn(1, 1, 224, 224)
