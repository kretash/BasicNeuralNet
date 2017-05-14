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
static double s_alpha = 0.1; // Momentum

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

   void _compute_error(double delta, double learning_rate );

	double m_weight = 0.0;
   double m_delta_weight = 0.0;
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

   // pushes a value to the node. When all links have been pushed
   // the node will fire the values forward.
   // at the begining of the network there will be no way to start as there are no back links.
   // once all back links have pushed the nodes will fire. If this is the first node it needs 
   // force start to true as there will be no back nodes.
	void _push(double value, bool force_start = false);
	double _sigmoid( double num );
	void _fire();

   // computes the error. similar to push but backwards.
   void _compute_error(double delta, double learning_rate, bool force_start = false );

	bool m_end_node = false;

   // The last value fired forward.
   double m_value = 0.0;
   // the sum of all the values of fired to this node.
	double m_synapse_sum = 0.0;

   double m_error_sum = 0.0;
	double m_delta = 0.0;
	double m_bias = 0.0;

   // When the load is equal to the number of back links the node will fire.
	int32_t m_forward_load = 0;
   // When the load is equal to the number of forward the node will fire back.
   // Calculating the error made from the previous iteration.
   int32_t m_backward_load = 0;

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