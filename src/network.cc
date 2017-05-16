#include <assert.h>
#include <iostream>

#include "network.hh"

#define FAKE_RANDOM 0

double random( double min, double max ) {
#if FAKE_RANDOM
	return min + (max - min)*0.75;
#else
	return min + static_cast < double > ( rand() ) / ( static_cast < double > ( RAND_MAX / ( max - min ) ) );
#endif
}

std::vector<double> Network::activate( std::vector<double> inputs ) {

	auto inodes = m_input_layer->m_nodes;

	assert( inputs.size() == inodes.size() &&
		"INPUT MUST BE THE SIZE OF THE NODES IN THE INPUT LAYER" );

	for( size_t i = 0; i < inodes.size(); ++i ) {
		inodes[i]->_input( inputs[i] );
	}

	std::vector<double> outputs;
	for( auto o : m_output_layer->m_nodes ) {
		outputs.push_back( o->m_value );
	}
	return outputs;
}

void Network::propagate( double learning_rate, std::vector<double> target ) {

	auto onodes = m_output_layer->m_nodes;

	assert( target.size() == onodes.size() &&
		"RESULTS MUST BE THE SIZE OF THE NODES IN THE OUTPUT LAYER" );

	for( size_t i = 0; i < onodes.size(); ++i ) {

		auto on = onodes[i];
		double this_error = target[i] - on->m_value;
		on->_start_compute_error( this_error, learning_rate );

	}

	for( size_t i = 0; i < onodes.size(); ++i ) {
		auto on = onodes[i];
		on->_start_backpropagate( learning_rate );
	}

}