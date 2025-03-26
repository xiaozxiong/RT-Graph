'''
Energy-Delay Product (EDP): 
    EDP = Energy * Delay
    Energy = Power * Time (Joules)
    EDP = Power * Time * Delay
'''
import pynvml
import time

# def get_gpu_power():
#     pynvml.nvmlInit()

#     handle = pynvml.nvmlDeviceGetHandleByIndex(0)
#     power_mW = pynvml.nvmlDeviceGetPowerUsage(handle)
#     power_W = power_mW / 1000.0  # Convert to watts

#     pynvml.nvmlShutdown()
#     return power_W

def get_gpu_power_over_time():
    pynvml.nvmlInit()

    handle = pynvml.nvmlDeviceGetHandleByIndex(1)

    while True:
        power_mW = pynvml.nvmlDeviceGetPowerUsage(handle)
        power_W = power_mW / 1000.0  # Convert to watts
        print(f'{power_W}')
        time.sleep(0.0001)  # Measure every milisecond

    pynvml.nvmlShutdown()   
    
if __name__ == '__main__':
    get_gpu_power_over_time()
