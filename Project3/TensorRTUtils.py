# TensorRTUtils
#!pip install pycuda
#!pip install tensorrt
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
import os

class MatrixIterator:
    """Class to implement an iterator on a matrix"""

    def __init__(self, matrix, n=0, max=0):
        self.matrix = matrix
        if max > 0:
          self.max    = max
        else:
          self.max    = matrix.shape[0]
        self.n      = n

    def __iter__(self):
        return self

    def __next__(self):
        if self.n <= self.max:
            result = self.matrix[self.n,:,:].squeeze()
            self.n += 1
            return result
        else:
            raise StopIteration

    def first(self):
        return self.matrix[0,:,:].squeeze()

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

class ErrorRecorder(trt.IErrorRecorder):
    def __init__(self):
        trt.IErrorRecorder.__init__(self)
        self.errorsStack = []

    def clear(self):
        self.errorsStack.clear()
    def get_error_code(self, arg0):
        #Error code saved in the error tuple first position
        return self.errorsStack[arg0][0]
    def get_error_desc(self, arg0):
        # Error code saved in the error tuple second position
        return self.errorsStack[arg0][1]
    def has_overflowed(self):
        return False
    def num_errors(self):
        return len(self.errorsStack)
    def report_error(self, arg0, arg1):
        error = (arg0, arg1)
        #Errors will be saved as a list of tuples, each tuple will be a pair of error code and error description
        self.errorsStack.append(error)

class Logger(trt.ILogger):
    def __init__(self):
        trt.ILogger.__init__(self)

    def log(self, severity, msg):
        if severity == trt.ILogger.INTERNAL_ERROR:
            print('INTERNAL_ERROR')
        elif severity == trt.ILogger.ERROR:
            print('TRT - ERROR')
        elif severity == trt.ILogger.WARNING:
            print('TRT - WARNING')
        elif severity == trt.ILogger.INFO:
            print('TRT - INFO')
        elif severity == trt.ILogger.VERBOSE:
            print('TRT - VERBOSE')
        else:
            print('TRT - Wrong severity')

        print(msg)

class Int8EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, calibrationSetPath = None, calibSet = None):
        # Whenever you specify a custom constructor for a TensorRT class,
        # you MUST call the constructor of the parent explicitly.
        trt.IInt8EntropyCalibrator2.__init__(self)

        self.cacheFile = calibrationSetPath + '/CacheFile.bin'
        self.batchSize = 1
        self.currentIndex = 0
        self.deviceInput = None
        self.currentIndex = 0
        self.PreProcessedSetPath = calibrationSetPath + '/PreProcessedSet'
        self.PreProcessedSetCount = calibSet.max
        self.PreProcessedSize = calibSet.first().size * 4 # float 32
        self.currentIndex = 0

        # Allocate enough memory for a whole batch.
        self.deviceInput = cuda.mem_alloc(self.PreProcessedSize)

        if os.path.exists(self.cacheFile):
            print('Calibration cache file already exists - ', self.cacheFile)
            return

        if os.path.isdir(self.PreProcessedSetPath):
            filesCnt = os.listdir(self.PreProcessedSetPath)

            if len(filesCnt) == self.PreProcessedSetCount:
                print('ERROR - Pre processed file set exists!!!')
                return
        else:
            os.makedirs(self.PreProcessedSetPath)

        if self.PreProcessedSetCount == 0:
            print('ERROR - Calibration set is empty!!!')

        print('Start calibration batches build')

        print(f"Nir: PreProcessedSetCount = {self.PreProcessedSetCount}") # Debug printing
        for idx in range(self.PreProcessedSetCount):
            preProcImg = next(calibSet)
            if idx % 100 == 0:
              print(f"Nir: {idx} preProcImg shape: {preProcImg.shape}") # Debug printing
            preProcessedFile = open(self.PreProcessedSetPath + '/' + str(idx) + '.bin', mode='wb')
            preProcImg.tofile(preProcessedFile)
            preProcessedFile.close()

        print('End calibration batches build')

    def get_algorithm(self):
        return trt.CalibrationAlgoType.ENTROPY_CALIBRATION_2

    def get_batch_size(self):
        return self.batchSize

    # TensorRT passes along the names of the engine bindings to the get_batch function.
    # You don't necessarily have to use them, but they can be useful to understand the order of
    # the inputs. The bindings list is expected to have the same ordering as 'names'.
    def get_batch(self, names):
        if not self.currentIndex < self.PreProcessedSetCount:
            return None

        print('Get pre processed file index - ', not self.currentIndex)

        batchData = np.fromfile(self.PreProcessedSetPath + '/' + str(self.currentIndex) + '.bin', dtype=np.single)

        cuda.memcpy_htod(self.deviceInput, batchData)
        self.currentIndex += 1

        return [self.deviceInput]

    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self.cacheFile):
            with open(self.cacheFile, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cacheFile, "wb") as f:
            f.write(cache)

logger = Logger()
errorRecorder = ErrorRecorder()

builder = trt.Builder(logger)
builder.max_batch_size = 1

calib = None
config = builder.create_builder_config()
config.max_workspace_size = 1073741824

optimizationProfiler = builder.create_optimization_profile()

networkFlags = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
network = builder.create_network(networkFlags)
parser = trt.OnnxParser(network, logger)
runtime = trt.Runtime(logger)

engine = None
context = None

modelName = None

inputs = []
outputs = []
bindings = []
stream = None

def TrtModelParse(modelPath):
    global modelName
    global parser
    global network

    modelName = modelPath.split('.')[0]
    parseResult = parser.parse_from_file(modelPath)

    if (not parseResult):
        for error in range(parser.num_errors):
            print(str(parser.get_error(error)))
    else:
        print("Model parsing OK!")

        print("Network Description")

        inputs = [network.get_input(i) for i in range(network.num_inputs)]
        outputs = [network.get_output(i) for i in range(network.num_outputs)]

        for input in inputs:
            print("Input '{}' with shape {} and dtype {}".format(input.name, input.shape, input.dtype))
        for output in outputs:
            print("Output '{}' with shape {} and dtype {}".format(output.name, output.shape, output.dtype))

def TrtModelOptimizeAndSerialize(precision = 'fp32',calibPath="", calibSet=None):
    global modelName
    global builder
    global optimizationProfiler
    global calib
    global config
    global network
    global engine
    global runtime

    global g_DEBUG_network
    global g_DEBUG_config
    global g_DEBUG_modelOptName

    modelOptName = modelName + precision + '.trt.engine'
    g_DEBUG_modelOptName = modelOptName

    if os.path.exists(modelOptName):
        with open(modelOptName, 'rb') as f:
            engine = runtime.deserialize_cuda_engine(f.read())
    else:
        inputs = [network.get_input(i) for i in range(network.num_inputs)]
        input = network.get_input(0)

        inputShape = [1, input.shape[1], input.shape[2], input.shape[3]]

        optimizationProfiler.set_shape(input.name, inputShape, inputShape, inputShape)

        config.add_optimization_profile(optimizationProfiler)

        if precision == 'fp16':
            if builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
        elif precision == 'int8':
            if builder.platform_has_fast_int8:
                if builder.platform_has_fast_fp16:
                    # Also enable fp16, as some layers may be even more efficient in fp16 than int8
                    config.set_flag(trt.BuilderFlag.FP16)

                config.set_flag(trt.BuilderFlag.INT8)

                calib = Int8EntropyCalibrator(calibPath, calibSet)
                config.int8_calibrator = calib

        g_DEBUG_network = network
        g_DEBUG_config  = config
        try:
            print("Nir: TrtModelOptimizeAndSerialize before build_serialized_network")
            engine = builder.build_serialized_network(network, config)
            print("Nir: TrtModelOptimizeAndSerialize after build_serialized_network")
        except AttributeError:
            print("Nir: TrtModelOptimizeAndSerialize before build_engine")
            non_serialized_engine = builder.build_engine(network, config)
            print("Nir: TrtModelOptimizeAndSerialize after build_engine & before serialize")
            engine = non_serialized_engine.serialize()
            print("Nir: TrtModelOptimizeAndSerialize after serialize")

        engineFD = open(modelOptName, 'wb')
        engineFD.write(engine)
        engineFD.close()

    #print('TRT engine - ', engine.device_memory_size, ' Bytes')
    #engineDeviceMemory = 0
    #engineDeviceMemory += engine.device_memory_size
    #print('TRT engine number of layers - ', engine.num_layers)
    #print('TRT engine number of bindings - ', engine.num_bindings)
    #print('TRT engine number of profils - ', engine.num_optimization_profiles)

    print('Completion optimized model')

def ModelInferSetup():
    global context
    global engine
    global inputs
    global outputs
    global bindings
    global stream

    stream = cuda.Stream()

    #Over all Tensors inputs & outputs of the TRT engine
    #TRT hold first all Tensors inputs and after the Tensor outptus
    for binding in engine:
        #Get current binded Tensor volume size in elemente units
        size = trt.volume(engine.get_binding_shape(binding))
        #Get current binded Tensor element type
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host page locked bbuffer
        host_mem = cuda.pagelocked_empty(size, dtype)
        # Allocate device bbuffer
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))

    # Contexts are used to perform inference.
    context = engine.create_execution_context()
    context.error_recorder = errorRecorder

def Inference(externalnputs = None):

    global context
    global stream
    global inputs
    global outputs
    global bindings

    try:
        #verify that TRT context generated successfully
        if context is not None:
            #Verify that inputs to inference are exist
            if externalnputs is not None:
                #Copy all Tensors inputs data from user memory to TRT host page locked memory before loading it to the device
                if len(externalnputs) == len(inputs):
                    for index in range(len(externalnputs)):
                        if len(inputs[index].host) == externalnputs[index].size:
                            np.copyto(inputs[index].host, externalnputs[index].ravel())
                        else:
                            print('TRT external input size - ', externalnputs[index].size,
                                  ' is not equal to model inputs size - ', len(inputs[index].host))
                            return None

                    # Transfer input data to the GPU from the host page locked memory.
                    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
                    # Run asynchronously inference using the user\internal stream.
                    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
                    # Transfer predictions back from the GPU.
                    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]

                    stream.synchronize()
                    # Build a list of Tensors outputs and return only the host outputs.
                    return [out.host for out in outputs]
                else:
                    print('External inputs list size - ', len(externalnputs), ' is not equal to model inputs list size - ', len(inputs))
                    return None
            else:
                print('External inputs list is None ERROR')
                return None
    except BaseException as e:
        msg = e
        print('TRT inference exception ERROR - ', msg)
