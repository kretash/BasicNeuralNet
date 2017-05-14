#include "network.hh"

Link::Link( std::shared_ptr<Node> from, std::shared_ptr<Node> to ) {
	m_link_from = from;
	m_link_to = to;
	m_weight = random( -1.0, 1.0 );
}

void Link::_push( double value ) {
	m_link_to->_push( value * m_weight );
}

void Link::_fire() {
	m_link_to->_fire();
}

void Link::_compute_error( double delta, double learning_rate ) {

	double this_delta = delta * m_weight;
	m_link_from->_compute_error( this_delta, learning_rate );
}

void Link::_backpropagate( double delta, double learning_rate ) {

	const double lamda = 0.0;

	m_weight += 
		learning_rate
		* ( delta * m_link_from->m_value + lamda * m_link_from->m_value )
		+ s_alpha * m_delta_weight;

	m_delta_weight = learning_rate * delta * m_link_from->m_value;
	m_link_to->m_bias += learning_rate * delta;

}