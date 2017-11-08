# Basic Neural Net

The most basic neural net you can build. Completly from scratch in C++.

This network currently trains and produces results for an XOR.

---

## Building the network

The first thing we do is create the different layers. We specify how many nodes each layer has.

```cpp
    std::shared_ptr<Layer> input_layer = std::make_shared<Layer>( 2, "Input Layer" );
    std::shared_ptr<Layer> hidden_layer = std::make_shared<Layer>( 3, "Hidden Layer" );
    std::shared_ptr<Layer> output_layer = std::make_shared<Layer>( 1 , "Output Layer" );
```

![image1]

The next step is to create the links between the layers and create a network. The function **project** will connect all the nodes in a layer to the layer passed as an argument.

```cpp
    input_layer->project( hidden_layer );
    hidden_layer->project( output_layer );
```

This is how our layers look now.

![image2]

 Finally, to create the network we need pass the input and output layers.
 
```cpp
    std::shared_ptr<Network> network = std::make_shared<Network>( input_layer, output_layer );
```

## Using the Network

To use the network we call the function **activate**. This function will take an std::vector as a parameter that should match in size with the number of nodes in the input layer. The return vector will be the same size as the number of output layers. 

```cpp
    std::vector<double> output = network->activate( { 0,0 } );
```

The output values we will get the first times we use the network will be random. In order to train the network we can call **propagate** after an activate to trains the network. We want to pass in a learning rate and the target results. The learning rate is just a constant defined to 0.5 and the target results must be an std::vector that matches in size with the amount of output nodes. The learning rate can be tweaked to achieve better results.

```cpp
    network->propagate( learning_rate, { 0 } );
```

Once this is done a multitude of times with all posible values the results from the activate will start to match the target results.

---

## Building

There is a Visual Studio solution file and xcode project that can be use to build. There is only a small number of files, so building from the command line would be easy too.

### License

MIT License

[image1]: https://github.com/kretash/BasicNeuralNet/raw/master/images/image1.png "Layers"
[image2]: https://github.com/kretash/BasicNeuralNet/raw/master/images/image2.png "Final Network"
