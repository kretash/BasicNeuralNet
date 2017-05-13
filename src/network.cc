#include <assert.h>

#include "network.hh"

double random( double min, double max ){
	return min + static_cast <double> ( rand() ) / ( static_cast <double> ( RAND_MAX / ( max - min ) ) );
}

std::vector<double> Network::activate(std::vector<double> inputs ){

	auto inodes = m_input_layer->m_nodes;

	assert( inputs.size() == inodes.size() &&
		"INPUT MUST BE THE SIZE OF THE NODES IN THE INPUT LAYER");

	for( size_t i = 0; i < inodes.size(); ++i ){
		inodes[i]->_push( inputs[i] );
	}

	for( size_t i = 0; i < inodes.size(); ++i ) {
		inodes[i]->_fire();
	}
		
	std::vector<double> outputs;
	for( auto o : m_output_layer->m_nodes ){
		outputs.push_back( o->m_synapse_sum );
	}
	return outputs;
}

void Network::propagate( double learning_rate, std::vector<double> results ) {

	auto onodes = m_output_layer->m_nodes;

	assert( results.size() == onodes.size() &&
		"RESULTS MUST BE THE SIZE OF THE NODES IN THE OUTPUT LAYER" );

	double error = 0.0;

	for( size_t i = 0; i < onodes.size(); ++i ){

		auto on = onodes[i];

		double this_error = results[i] - on->m_synapse_sum;
		on->m_delta = (1.0 - on->m_synapse_sum) * results[i] * this_error;
		error += (0.5 * this_error * this_error );
		
	}

}