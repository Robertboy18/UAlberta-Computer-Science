// simplified view: some members have been omitted,
// and some signatures have been altered

// helpful typedef's

typedef std::vector< NNLayer* >  VectorLayers;
typedef std::vector< NNWeight* >  VectorWeights;
typedef std::vector< NNNeuron* >  VectorNeurons;
typedef std::vector< NNConnection > VectorConnections;


// Neural Network class

class NeuralNetwork  
{
public:
    NeuralNetwork();
    virtual ~NeuralNetwork();
    
    void Calculate( double* inputVector, UINT iCount, 
        double* outputVector = NULL, UINT oCount = 0 );

    void Backpropagate( double *actualOutput, 
         double *desiredOutput, UINT count );

    VectorLayers m_Layers;
};


// Layer class

class NNLayer
{
public:
    NNLayer( LPCTSTR str, NNLayer* pPrev = NULL );
    virtual ~NNLayer();
    
    void Calculate();
    
    void Backpropagate( std::vector< double >& dErr_wrt_dXn /* in */, 
        std::vector< double >& dErr_wrt_dXnm1 /* out */, 
        double etaLearningRate );

    NNLayer* m_pPrevLayer;
    VectorNeurons m_Neurons;
    VectorWeights m_Weights;
};


// Neuron class

class NNNeuron
{
public:
    NNNeuron( LPCTSTR str );
    virtual ~NNNeuron();

    void AddConnection( UINT iNeuron, UINT iWeight );
    void AddConnection( NNConnection const & conn );

    double output;

    VectorConnections m_Connections;
};


// Connection class

class NNConnection
{
public: 
    NNConnection(UINT neuron = ULONG_MAX, UINT weight = ULONG_MAX);
    virtual ~NNConnection();

    UINT NeuronIndex;
    UINT WeightIndex;
};


// Weight class

class NNWeight
{
public:
    NNWeight( LPCTSTR str, double val = 0.0 );
    virtual ~NNWeight();

    double value;
};