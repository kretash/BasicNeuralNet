#include "network.hh"

Layer::Layer( int32_t nodes ) {
	for( int32_t i = 0; i < nodes; ++i ) {
		std::shared_ptr<Node> n = std::make_shared<Node>();
		m_nodes.push_back( n );
	}
}

void Layer::project( std::shared_ptr<Layer> other_layer ) {
	for( auto i_node : m_nodes ) {
		for( auto o_node : other_layer->m_nodes ) {

			std::shared_ptr<Link> link = std::make_shared<Link>( i_node, o_node );
			i_node->add_f_link( link );
			o_node->add_b_link( link );

		}
	}
}
