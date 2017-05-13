/*
	Defines Network, Layer, Node and Link
	Author: Carlos Martinez Romero (kretash)
	License: MIT License
*/

#include <memory>
#include <vector>

double random( double min, double max);

class Node;

static int32_t s_component_id_count = 0;
class Component{
public:
	Component(){
		m_component_id = ++s_component_id_count;
	}
	~Component() {

	}
	int32_t m_component_id = 0;
};

class Link : public Component {
protected:
	Link() {}
public:
	Link( std::shared_ptr<Node> from, std::shared_ptr<Node> to );
	~Link() {}

private:
	friend class Node;

	void _push( double value );
	void _fire();

	double m_weight = 0.0;
	std::shared_ptr<Node> m_link_to;
	std::shared_ptr<Node> m_link_from;
};

class Node : public Component {
public:
	Node() {
		m_bias = random(0.0, 1.0);
	};
	~Node() {};

	void add_f_link( std::shared_ptr<Link> link );
	void add_b_link( std::shared_ptr<Link> link );

private:
	friend class Network;
	friend class Link;

	void _push(double value);
	double _sigmoid( double num );
	void _fire();

	bool m_end_node = false;
	double m_synapse_sum = 0.0;
	double m_delta = 0.0;
	double m_bias = 0.0;
	int32_t m_load = 0;
	std::vector<std::shared_ptr<Link>> m_f_links;
	std::vector<std::shared_ptr<Link>> m_b_links;

};

class Layer : public Component {
protected:
	Layer() {}

public:

	Layer( int32_t nodes );
	~Layer() {}

	void project( std::shared_ptr<Layer> other_layer );

private:
	friend class Network;

	std::vector<std::shared_ptr<Node>> m_nodes;
};

class Network : public Component {
protected:
	Network() {}
public:

	Network( std::shared_ptr<Layer> input, std::shared_ptr<Layer> output ) :
		m_input_layer( input ), m_output_layer( output ) {
		for( auto n : m_output_layer->m_nodes ){
			n->m_end_node = true;
		}
	}

	~Network() {}

	std::vector<double> activate( std::vector<double> inputs );
	void propagate( double learning_rate, std::vector<double> results );

private:
	std::shared_ptr<Layer> m_input_layer;
	std::shared_ptr<Layer> m_output_layer;

};