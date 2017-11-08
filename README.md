# BasicNeuralNet

The most basic neural net you can build. Completlly from scratch in C++.

The first thing we do is create the different layers. We specity how many nodes each layer has.

```cpp
    std::shared_ptr<Layer> input_layer = std::make_shared<Layer>( 2, "Input Layer" );
    std::shared_ptr<Layer> hidden_layer = std::make_shared<Layer>( 3, "Hidden Layer" );
    std::shared_ptr<Layer> output_layer = std::make_shared<Layer>( 1 , "Output Layer" );
```

![image1]

The next step is to create the links between the layers and create a network. The function **project** will connect all the nodes in a layer to the layer pased as an argument.

```cpp
    input_layer->project( hidden_layer );
    hidden_layer->project( output_layer );
```

This is how our network looks now.

![image2]

 Finally, to create the network we need pass the input and output layers.
 
```cpp
    std::shared_ptr<Network> network = std::make_shared<Network>( input_layer, output_layer );
```

[image1]: https://github.com/kretash/BasicNeuralNet/raw/master/images/image1.png "Layers"
[image2]: https://github.com/kretash/BasicNeuralNet/raw/master/images/image2.png "Final Network"
