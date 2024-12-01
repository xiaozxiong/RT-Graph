import os

os.chdir("./bin")

# q = 2 ^ 24
for i in range(0,28,1):
    data_size = (1<<i)
    os.system("./measure -d "+str(data_size)+" --device 3")

# q = 2 ^ 16
for i in range(0,28,1):
    data_size = (1<<i)
    os.system("./measure -d "+str(data_size)+" -q 65536 --device 3")

# no duplicated elements
# for i in range(0,25,1):
#     data_size = (1<<i)
#     os.system("./measure -d "+str(data_size)+" --device 3")

# 2^14 = 16384
# 2^15 = 32768
# 2^16 = 65536
# 2^20 = 1048576
# 2^24 = 16777216