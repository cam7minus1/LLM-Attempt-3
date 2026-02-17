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
