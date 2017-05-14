#include "network.hh"

void Node::add_f_link( std::shared_ptr<Link> link ){
	m_f_links.push_back( link );
}

void Node::add_b_link( std::shared_ptr<Link> link ) {
	m_b_links.push_back( link );
}

void Node::_push( double value , bool force_start) {
	m_synapse_sum += value;
	++m_forward_load;
	if( m_forward_load == m_b_links.size() || force_start){
		_fire();
	}
}

// transfer function
double  Node::_sigmoid( double num ) {
	return ( 1.0 / ( 1.0 + exp( -num ) ) );
}

void Node::_fire( ){

	double v = _sigmoid( m_synapse_sum + m_bias);
   m_value = v;

	if( m_end_node )	return;

	for( auto link : m_f_links ){
		link->_push(v);
	}
	
	m_synapse_sum = 0;
   m_forward_load = 0;
}

void Node::_compute_error(double delta, double learning_rate, bool force_start ){

   m_error_sum += delta;

   ++m_backward_load;
   if( m_backward_load == m_f_links.size() || force_start ){

      m_delta = ( 1.0 - m_value ) * m_value * delta;

      for( auto blinks : m_b_links ){
         blinks->_compute_error(m_delta, learning_rate);
      }

      m_error_sum = 0.0;
      m_backward_load = 0;
   }
}