import ctypes
so = ctypes.CDLL('./libkernel.so')
so.run()