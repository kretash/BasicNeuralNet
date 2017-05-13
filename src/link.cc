#include "network.hh"

Link::Link( std::shared_ptr<Node> from, std::shared_ptr<Node> to ){
	m_link_from = from;
	m_link_to = to;
	m_weight = random(-1.0, 1.0);
}

void Link::_push( double value ) {
	m_link_to->_push( value * m_weight );
}

void Link::_fire() {
	m_link_to->_fire();
}