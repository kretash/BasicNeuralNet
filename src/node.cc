#include "network.hh"

void Node::add_f_link( std::shared_ptr<Link> link ) {
	link->m_name += "(" + this->m_name + " -> link forward)";
	m_f_links.push_back( link );
}

void Node::add_b_link( std::shared_ptr<Link> link ) {
	link->m_name += "(" + this->m_name + " -> <- link back)";
	m_b_links.push_back( link );
}

void Node::_input( double value ){

	m_value = value;

	for( auto link : m_f_links ) {
		link->_push( m_value );
	}

	m_synapse_sum = 0;
	m_forward_load = 0;
}

void Node::_push( double value ) {
	m_synapse_sum += value;
	++m_forward_load;
	if( m_forward_load == m_b_links.size() ) {
		_fire();
	}
}

// transfer function
double  Node::_sigmoid( double num ) {
	return ( 1.0 / ( 1.0 + exp( -num ) ) );
}

void Node::_fire() {

	m_value = _sigmoid( m_synapse_sum + m_bias );

	m_synapse_sum = 0;
	m_forward_load = 0;

	if( m_end_node ) return;

	for( auto link : m_f_links ) {
		link->_push( m_value );
	}
}

void Node::_start_compute_error( double delta, double learning_rate ) {

	m_delta = ( 1.0 - m_value ) * m_value * delta;

	m_error_sum = 0.0;
	m_backward_load = 0;

	for( auto blinks : m_b_links ) {
		blinks->_compute_error( m_delta, learning_rate );
	}
}

void Node::_compute_error( double delta, double learning_rate ) {

	m_error_sum += delta;

	++m_backward_load;
	if( m_backward_load == m_f_links.size() ) {

		m_delta = ( 1.0 - m_value ) * m_value * m_error_sum;

		m_error_sum = 0.0;
		m_backward_load = 0;

		for( auto blinks : m_b_links ) {
			blinks->_compute_error( m_delta, learning_rate );
		}
	}
}

void Node::_start_backpropagate( double learning_rate ) {
	
	m_bias += learning_rate * m_delta;

	m_backward_load = 0;

	for( auto blinks : m_b_links ) {
		blinks->_backpropagate( learning_rate );
	}
}

void Node::_backpropagate( double learning_rate ) {

	++m_backward_load;
	if( m_backward_load == m_f_links.size() ) {

		m_bias += learning_rate * m_delta;

		m_backward_load = 0;

		for( auto blinks : m_b_links ) {
			blinks->_backpropagate( learning_rate );
		}

	}
}