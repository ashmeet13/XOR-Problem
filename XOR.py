from copy import copy,deepcopy
import numpy as np
import math

'''
Sigmoid:
    Returns: Sigmoid of a number
'''

def sigmoid(value):
    try:
        value = float(math.exp(-value))
        value = float(value + 1)
        value = float(1/value)
        return value
    except:
        print(value)

'''
Sigmoid Derivatve:
    Returns: Sigmoid Derivative of a number
'''

def sigmoid_derivative(value):
    return (value*(1-value))

'''
Hypothesis:
    Returns: Final value after vector multiplication of weights and inputs
'''

def hypothesis(theta, x):
    return(sum(np.multiply(theta,x)))

'''
Forward Pass:
    This function is used to calculate the forward pass of
    the neural network
'''

def forward_pass(features,train_parameters):
    hidden_activations = []
    hidden = np.dot(features,train_parameters[0])
    hidden = [1.0]+list(map(sigmoid,hidden))
    hidden_activations.append(np.asarray(hidden))

    total_layers = len(train_parameters[1])+1
    for layer in range(1,total_layers):
        activation = hidden_activations[layer-1]
        current_parameter = train_parameters[1][layer-1]
        hidden = np.dot(activation,current_parameter)
        hidden = [1.0]+list(map(sigmoid,hidden))
        hidden_activations.append(np.asarray(hidden))

    output = sigmoid(np.dot(hidden_activations[total_layers-1],train_parameters[2]))
    return output,hidden_activations

'''
Train:
    Will control the entire process of forward and backward pass
    for the neural network
'''

def train(train_input, train_output, train_parameters, learning_rate,delta):
    batch_delta = deepcopy(delta)
    batch_run = 0
    batch_size = len(train_input)
    for index in range(len(train_input)):
        features = train_input[index]
        error = []
        total_layers = len(train_parameters[1])+1
        hidden_activations = []
        
        '''
        Forward Pass
        '''
        output,hidden_activations = forward_pass(features,train_parameters)
        '''
        Final Activations Recieved after forward pass
        '''


        '''
        Backward Pass Starts
        '''

        '''
        > Calculates the loss between the desired output
        > and the predicted output
        '''

        flag = 0
        error.append((output-train_output[index]))


        activation = hidden_activations[total_layers-1]
        parameter = train_parameters[2]
        parameter = (error[0]*parameter)
        derivative = list(map(sigmoid_derivative,activation))
        derivative = np.asarray(derivative)
        add_error = derivative*parameter
        add_error = add_error[1:]
        error.append(add_error)

        error_flag = 1
        parameter_flag = len(train_parameters[1])-1
        
        
        for activate in range(total_layers-2,-1,-1):
            add_error = []
            activation = hidden_activations[activate]
            derivative = list(map(sigmoid_derivative,activation)) # 17
            parameter_list = train_parameters[1][parameter_flag]  # 17x16
            current_error = error[error_flag]     #16
            for run in range(len(derivative)):
                add_error.append((derivative[run])*np.dot(parameter_list[run],current_error))
            error.append(np.asarray(add_error)[1:])
            parameter_flag = parameter_flag - 1
            error_flag = error_flag + 1
        error = error[::-1]

        '''
        >Errors in every layer of the neural network recieved
        '''
            
        '''
        > Calculation of delta function for the neural network
        '''

        for feature_index in range(len(features)):
            batch_delta[0][feature_index]+= (features[feature_index]*error[0])

        error = error[1:]
        for layer in range(total_layers-1):
            activation = hidden_activations[layer]  #17
            for delta_index in range(len(batch_delta[1][0])):
                batch_delta[1][layer][delta_index] += (activation[delta_index]*error[layer])
        batch_delta[2] += hidden_activations[total_layers-1]*float(error[-1])

        '''
        > Calculated the delta for every layer in the neural networl
        '''
 
    '''
    > Calibrating the weights using the delta loss initiated
    '''

    for i in range(len(batch_delta)):
        batch_delta[i] = np.asarray(batch_delta[i])
        batch_delta[i]=batch_delta[i]/batch_size

    for p_index in range(len(train_parameters[0])):
        train_parameters[0][p_index] -= (learning_rate*batch_delta[0][p_index])

    for h_index in range(len(train_parameters[1])):
        for p_index in range(len(train_parameters[1][h_index])):
            train_parameters[1][h_index][p_index] -= (learning_rate*batch_delta[1][h_index][p_index])
    train_parameters[2] -= (learning_rate*batch_delta[2])
    batch_delta = deepcopy(delta)

    '''
    > Weights Calibrated
    '''

    '''
    Backward Pass Ends
    '''

    return train_parameters

'''
Driver:
    Initiate's every new epoch of the neural network
'''

def driver(train_input, train_output, train_parameters, delta):

    epochs = 5000
    learning_rate = 0.5
    for epoch in range(epochs+1):
        train_parameters = train(train_input, train_output, train_parameters, learning_rate, delta)

        if epoch%1000 == 0:
            print("Epoch :",epoch)
            predicted = []
            for test_index in range(4):
                features = train_input[test_index]
                output,hidden_activations = forward_pass(features, train_parameters)
                if output>=0.5:
                    predicted.append(1)
                else:
                    predicted.append(0)
            print("Desired Output: ",[0,1,1,0])
            print("Predicted Output: ",predicted)
            print()
            print("--------------------------------------")
            print()
            
    return train_parameters

'''
Main:
    Initiates the entire code by initializing random
    weights for every node in every hidden layer
'''

def main():

    print("Following output is expected for the input")
    print("Input - 0 0 , Output - 0")
    print("Input - 0 1 , Output - 1")
    print("Input - 1 0 , Output - 1")
    print("Input - 1 1 , Output - 0")
    print()

    train_input =[[1,0,0],
                  [1,0,1],
                  [1,1,0],
                  [1,1,1]]

    train_output = [0,1,1,0]

    hidden_layers = 2
    nodes_per_layer = 4
    train_parameters = []
    delta = []

    '''
    Intitializing random wights for every
    node in the hidden layer
    '''

    input_parameter = np.random.rand(len(train_input[0]),nodes_per_layer)
    input_delta = np.zeros((len(train_input[0]),nodes_per_layer))

    hidden_parameter = []
    hidden_delta = []
    for run in range(hidden_layers-1):
        add_layer = np.random.rand(nodes_per_layer+1,nodes_per_layer)
        add_delta = np.zeros((nodes_per_layer+1,nodes_per_layer))
        hidden_parameter.append(add_layer)
        hidden_delta.append(add_delta)

    output_parameter = np.random.rand(nodes_per_layer+1)
    output_delta = np.zeros(nodes_per_layer+1)

    train_parameters.append(input_parameter)
    train_parameters.append(hidden_parameter)
    train_parameters.append(output_parameter)

    delta.append(input_delta)
    delta.append(hidden_delta)
    delta.append(output_delta)

    '''
    Weights Initialized
    '''

    train_parameters = driver(train_input, train_output, train_parameters, delta)


if __name__ == "__main__":
    main()
