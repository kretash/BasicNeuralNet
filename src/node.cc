#include "network.hh"

void Node::add_f_link( std::shared_ptr<Link> link ){
	m_f_links.push_back( link );
}

void Node::add_b_link( std::shared_ptr<Link> link ) {
	m_b_links.push_back( link );
}

void Node::_push( double value ) {
	m_synapse_sum += value;
	++m_load;
	if( m_load == m_b_links.size() ){
		_fire();
	}
}

// transfer function
double  Node::_sigmoid( double num ) {
	return ( 1.0 / ( 1.0 + exp( -num ) ) );
}

void Node::_fire( ){
	if( m_end_node ){
		return;
	}

	double v = _sigmoid( m_synapse_sum + m_bias);

	for( auto link : m_f_links ){
		link->_push(v);
	}
	
	m_synapse_sum = 0;
	m_load = 0;
}