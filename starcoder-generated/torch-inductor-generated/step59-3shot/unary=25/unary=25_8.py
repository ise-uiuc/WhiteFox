
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
        self.negative_slope = negative_slope
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 > 0
        v3 = v1 * self.negative_slope.to(v2.dtype)
        v4 = torch.where(v2, v1, v3)
        return v4

# Generating models for the given requirements on CUDA devices with the provided parameter ranges
def run_generate_and_verify_models(device_type, devices, negative_slope_range, verify_func):
    device_models = torch.jit.CompilationUnit()._initial_method_call(torch.device(device_type), None, None)

    def generate_model(neg_slope):
        m = Model(neg_slope)
        m.eval()
        m.to(device_models)
        m = torch.jit.script(m)
        m.to(device_models)
        m = torch.jit._recursive.wrap_cpp_module(m._c)
        device_models[neg_slope] = m
        return m

    num_iter = 0
    with torch.no_grad():
        verified = False
        while not verified:
            # Generate the model
            rand_idx = random.randrange(len(negative_slope_range))
            if num_iter < 5:
                neg_slope = negative_slope_range[rand_idx-3]
            else:
                neg_slope = negative_slope_range[rand_idx]
            m = generate_model(neg_slope)
            
            # Initialize the model
            with torch.random.fork_rng():
                # This seems to be required to get the model to work on both CPU and CUDA
                torch.random.manual_seed(12345)
                for di in device_type:
                    for device in devices:
                        try:
                            input = torch.randn(1, 3, 64, 64, device=device)
                            m(input)
                            failed = False
                        except Exception as e:
                            if 'out of range' not in str(e):
                                failed = True
                                if "out of CUDA memory" not in str(e):
                                    raise
                                log.info('\033[1A\033[KModel {} (negative slope {}) failed with:\n{}'.format(rand_idx, neg_slope, str(e)))
                
                num_iter += 1
            
            if num_iter > 5:
                break
    
    if num_iter > 1:
        print()

    verify_func(device_models, 'cuda', device_type, devices, neg_slope)

# Verify that the generated model works on the specified devices
def verify_models(device_models, expected_name, device_type, devices, neg_slope):
    with torch.no_grad():
        # Check each device
        for di in device_type:
            for i in range(len(devices)):
                device = devices[i]
                input = torch.randn(1, 3, 64, 64, device=device)
    
                # Print the model graph
                if device == 'cuda':
                    print('Model for {} (negative slope {}):\n{}'.format(expected_name, neg_slope, m.get_debug_state().str(20, True)))
                    print('Input:\n{}\n{}  ----------------- /  \033[F\033[K'.format(input, '\U000029BB' * (len(devices)-i)))
                else:
                    log.debug('Model for {} (negative slope {}):\n{}'.format(expected_name, neg_slope, m.get_debug_state().str(20, True)))
                    log.debug('Input:\n{}\n{}  ----------------- /  \033[F\033[K'.format(input, '\U000029BB' * (len(devices)-i)))
    
                # Test execution on specified device
                device_model = device_models[neg_slope].to(device)
                device_model(input)
        print()

# Run with no CUDA devices
device_type = []
devices = []
with_cuda = False
with_mkldnn = False
for available_device in ['cpu', 'cuda']:
    if available_device == 'cuda':
        device_type.append(available_device)
        devices.append(available_device)
        if torch.cuda.is_available():
            with_cuda = True
        else:
            log.info('CUDA device is available but not available.')
run_generate_and_verify_models(device_type, devices, negative_slope_range, lambda m, expected_name, device_type, devices, neg_slope: verify_models(m, expected_name, device_type, devices, neg_slope))

# Run with no CUDA devices
device_type = []
devices = []
with_cuda = False
with_mkldnn = False
for available_device in ['cpu', 'cuda']:
    if available_device == 'cuda':
        device_type.append(available_device)
        devices.append(available_device)
        if torch.cuda.is_available():
            with_cuda = True
        else:
            log.info('CUDA device is available but not available.')
run_generate_and_verify_models(device_type, devices, negative_slope_range, lambda m, expected_name, device_type, devices, neg_slope: verify_models(m, expected_name, device_type, devices, neg_slope))

# Run with no CUDA devices
device_type = []
devices = []
with_cuda = False
with_mkldnn = False
for available_device in ['cpu', 'cuda']:
    if available_device == 'cuda':
        device_type.append(available_device)
        devices.append(available_device)
        if torch.cuda.is_available():
            with_cuda = True
        else:
            log.info('CUDA device is available but not available.')
run_generate_and_verify_models(device_type, devices, negative_slope_range, lambda m, expected_name, device_type, devices, neg_slope: verify_models(m, expected_name, device_type, devices, neg_slope))

# Run with no CUDA devices
device_type = []
devices = []
with_cuda = False
with_mkldnn = False
for available_device in ['cpu', 'cuda']:
    if available_device == 'cuda':
        device_type.append(available_device)
        devices.append(available_device)
        if torch.cuda.is_available():
            with_cuda = True
        else:
            log.info('CUDA device is available but not available.')
run_generate_and_verify_models(device_type, devices, negative_slope_range, lambda m, expected_name, device_type, devices, neg_slope: verify_models(m, expected_name, device_type, devices, neg_slope))

# Run with CUDA devices
device_type = ['cuda']
devices = []
with_cuda = False
with_mkldnn = False
if with_cuda:
    for available_device in [0, 1]:
        devices.append('cuda:{}'.format(available_device))
    run_generate_and_verify_models(device_type, devices, negative_slope_range, lambda m, expected_name, device_type, devices, neg_slope: verify_models(m, expected_name, device_type, devices, neg_slope))
else:
    log.info('CUDA device is available but not available.')

# Run with the first CPU device
device_type = ['cpu']
devices = ['cpu']
with_cuda = False
with_mkldnn = False
run_generate_and_verify_models(device_type, devices, negative_slope_range, lambda m, expected_name, device_type, devices, neg_slope: verify_models(m, expected_name, device_type, devices, neg_slope))

# Run with the first and second CPU devices
device_type = ['cpu']
devices = ['cpu']
with_cuda = False
with_mkldnn = False
run_generate_and_verify_models(device_type, devices, negative_slope_range, lambda m, expected_name, device_type, devices, neg_slope: verify_models(m, expected_name, device_type, devices, neg_slope))

# Run with the first and third CPU devices
device_type = ['cpu']
devices = ['cpu']
with_cuda = False
with_mkldnn = False
run_generate_and_verify_models(device_type, devices, negative_slope_range, lambda m, expected_name, device_type, devices, neg_slope: verify_models(m, expected_name, device_type, devices, neg_slope))

# Run with the second and third CPU devices
device_type = ['cpu']
devices = ['cpu']
with_cuda = False
with_mkldnn = False
run_generate_and_verify_models(device_type, devices, negative_slope_range, lambda m, expected_name, device_type, devices, neg_slope: verify_models(m, expected_name, device_type, devices, neg_slope))

# Run with all three CPU devices
device_type = ['cpu']
devices = ['cpu']
with_cuda = False
with_mkldnn = False
run_generate_and_verify_models(device_type, devices, negative_slope_range, lambda m, expected_name, device_type, devices, neg_slope: verify_models(m, expected_name, device_type, devices, neg_slope))

print('Successfully verified that the models work.')