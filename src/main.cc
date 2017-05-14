#include <memory>
#include <iostream>
#include <time.h>

#include "network.hh"

int main( int argc, char** argv ) {

	srand((uint32_t)time(nullptr));

	// create all the resources
	std::shared_ptr<Layer> input_layer = std::make_shared<Layer>( 2 );
	std::shared_ptr<Layer> hidden_layer = std::make_shared<Layer>( 3 );
	std::shared_ptr<Layer> output_layer = std::make_shared<Layer>( 1 );

	input_layer->project( hidden_layer );
	hidden_layer->project( output_layer );

	std::shared_ptr<Network> network = std::make_shared<Network>( input_layer, output_layer );

	// train the network - learn XOR
	double learning_rate = 0.3;
	for( uint32_t i = 0; i < 10000; ++i ) {

		network->activate( { 0,0 } );
		network->propagate( learning_rate, { 0 } );

		network->activate( { 0,1 } );
		network->propagate( learning_rate, { 1 } );

		network->activate( { 1,0 } );
		network->propagate( learning_rate, { 1 } );

		network->activate( { 1,1 } );
		network->propagate( learning_rate, { 0 } );
	}

	std::cout << " 0 XOR 0 -> " << network->activate( { 0,0 } )[0] << "\n";
	std::cout << " 0 XOR 1 -> " << network->activate( { 0,1 } )[0] << "\n";
	std::cout << " 1 XOR 0 -> " << network->activate( { 1,0 } )[0] << "\n";
	std::cout << " 1 XOR 1 -> " << network->activate( { 1,1 } )[0] << "\n";

	return 0;
}
