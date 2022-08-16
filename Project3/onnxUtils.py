import json
import time
import tf2onnx
import onnx
import onnxsim
import os.path


# Save model into h5 and ONNX formats
def convertKerasToONNX(name, model, overwrite_existing = False):
    modelFile = name + '.onnx'
    if not os.path.isfile(modelFile) or overwrite_existing:
        # Save model with ONNX format
        (onnx_model_proto, storage) = tf2onnx.convert.from_keras(model)
        with open(os.path.join(modelFile), "wb") as f:
            f.write(onnx_model_proto.SerializeToString())
            f.close()
    
    return modelFile, onnx_model_proto, storage

def ModelOnnxCheck(name):

    msg = 'OK'
    isCheckOk = True

    print("===============================================================")
    print("Onnx model check report:")

    try:
        # Perform basic check on the model input
        onnx.checker.check_model(name + '.onnx')
        isCheckOk = True
    except onnx.checker.ValidationError as e:
        msg = e
        isCheckOk=False
    except BaseException as e:
        msg = e
        isCheckOk=False

    if isCheckOk:
        print('Model check completed Successfully')
    else:
        print('ERROR - Model check failure')

    print('Model onnx checker, check model - ', msg)

    return isCheckOk

def RemoveInitializerFromInput(model, modelPath):
    modelGraphInputs = model.graph.input
    startInputsCount = len(modelGraphInputs)

    nameToInput = {}
    for input in modelGraphInputs:
        nameToInput[input.name] = input

    for initializer in model.graph.initializer:
        if initializer.name in nameToInput:
            modelGraphInputs.remove(nameToInput[initializer.name])

    endInputsCount = len(modelGraphInputs)

    if startInputsCount != endInputsCount:
        print('Model includes several Initializers which considered as inputs to the graph - ', startInputsCount - endInputsCount)
        print('All Initializers were removed from graph inputs')
        print('Replace the model *.onx file with the updated one')
        onnx.save(model, modelPath)

def ProcessModelInputs(model, modelPath):
    RemoveInitializerFromInput(model, modelPath)
    modelGraphInputs = model.graph.input

    modelInputsDims = {}
    modelDynamicInputsDict = {}
    modelInputs = modelGraphInputs
    modelInputsNames = []
    print(str(modelInputs))

    for tensorInput in modelInputs:
        isInputDynamic = False
        modelDynamicInputShape = []
        for dim in tensorInput.type.tensor_type.shape.dim:
            if dim.dim_value == 0:
                isInputDynamic = True
                print('CAUTION!!! - Tensor input name' + ' - ', tensorInput.name, ', dimension - ' , dim.dim_param, ', set its value to 1 for Onnx simplify operation')
                modelDynamicInputShape.append(1)
            else:
                modelDynamicInputShape.append(dim.dim_value)

        modelInputsNames.append(tensorInput.name)

        if isInputDynamic is True:
            modelDynamicInputsDict[tensorInput.name] = modelDynamicInputShape

    return modelDynamicInputsDict

def ModelSimplify(name):

    msg = 'OK'
    nameSimp = name + 'Simp'
    model = None
    isSimplifiedOK = True

    if os.path.exists(nameSimp + '.onnx'):
        print('Model Onnx simplify is already exist, No model check and\or simplify operations is required')
        model = onnx.load(nameSimp + '.onnx')
        isSimplifiedOK = True
    else:
        print("===============================================================")
        print("Onnx model simplifier report:")
        model = onnx.load(name + '.onnx')

        modelDynamicInputsDict = ProcessModelInputs(model, name + '.onnx')

        try:
            print('Start model onnx simplify...')
            # Perform simplification on the model input
            model, check = onnxsim.simplify(model,input_shapes=modelDynamicInputsDict,
                                                  dynamic_input_shape=(len(modelDynamicInputsDict) > 0))
            print('Completion model onnx simplify')
            if (check):
                isSimplifiedOK = True
                print('Onnx simplification success!')
                print('Save Onnx simplified model to - ', nameSimp + '.onnx')
                onnx.save(model, nameSimp + '.onnx')
            else:
                isSimplifiedOK = False
                print('Onnx simplification failure!')
                print('Simplified Onnx model could not be generated and validated')
        except BaseException as e:
            print('Onnx simplification exception - ', e)
