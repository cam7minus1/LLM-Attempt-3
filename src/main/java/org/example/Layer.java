package org.example;

public class Layer {

    Neuron[] neurons;
    int weights;
    boolean isOutputLayer;

    // When conceptualizing this, remember the number of weights
    // per neuron is equal to the amount of nodes (or inputs) of the previous layer
    // LAYERS SHOULD USUALLY NOT HAVE THE SAME AMOUNT OF PARAMS!
    public Layer(int numNeurons, int numParams, boolean isOutputLayer){

        neurons = new Neuron[numNeurons];

        this.isOutputLayer = isOutputLayer;

        // Initialize a list a of neurons they will get default random weights
        for (int i = 0; i < numNeurons; i ++){
            neurons[i] = new Neuron(numParams, this.isOutputLayer);
        }

    }

    public float[] forwardPass(float[] input){
        float[] results = new float[neurons.length];
        for (int i = 0; i < neurons.length; i++){
            results[i] = neurons[i].forwardPass(input);
        }

        return results;
    }

    public float[] backwardsPass(float[] lastErrorSignals, float learningRate) {

        int numNeurons = neurons.length;
        int inputSize = neurons[0].getWeights().length;

        float[] prevErrorSignals = new float[inputSize];

        // For each neuron in this layer
        for (int n = 0; n < numNeurons; n++) {

            float blame = lastErrorSignals[n];

            // Apply ReLU derivative
            if (neurons[n].lastActivation <= 0) {
                blame = 0;
            }

            // Accumulate error for previous layer
            float[] w = neurons[n].getWeights();
            for (int i = 0; i < inputSize; i++) {
                prevErrorSignals[i] += w[i] * blame;
            }

            // Update weights + bias
            neurons[n].backwardsPass(blame, learningRate);
        }

        return prevErrorSignals;
    }

    @Override
    public String toString(){
        String result = "Layer - size:"+this.neurons.length+" \n";

        for (int i = 0; i<neurons.length; i++){
            float[] w = neurons[i].getWeights();
            result += "["+i+"] ";
            result += "b="+neurons[i].getBias()+" ";
            for(int j = 0; j < w.length; j++){
                result += "w"+j+"="+ w[j]+" ";
            }
            result += "\n";
        }

        return result;
    }
}
