package org.example;

public class Layer {

    Neuron[] neurons;
    int numNeurons;
    int inputSize;

    float[] lastErrorSignals;

    public Layer(int numNeurons, int inputSize, boolean isOutputLayer){
        this.numNeurons = numNeurons;
        this.inputSize = inputSize;
        this.neurons = new Neuron[numNeurons];
        this.lastErrorSignals = new float[numNeurons];

        for (int i = 0; i < numNeurons; i++){
            neurons[i] = new Neuron(inputSize, isOutputLayer);
        }
    }

    public float[] forwardPass(float[] inputs){
        float[] outputs = new float[numNeurons];

        for (int i = 0; i < numNeurons; i++){
            outputs[i] = neurons[i].forwardPass(inputs);
        }

        return outputs;
    }

    public float[] backwardsPass(float[] errorSignals, float learningRate){
        this.lastErrorSignals = errorSignals;

        float[] prevErrorSignals = new float[inputSize];

        for (int n = 0; n < numNeurons; n++) {

            float blame = lastErrorSignals[n];

            // Only apply ReLU derivative if NOT output layer
            if (!neurons[n].isOutputLayer && neurons[n].lastActivation <= 0) {
                blame = blame * 0.01f;
            }

            float[] w = neurons[n].getWeights();
            for (int i = 0; i < inputSize; i++) {
                prevErrorSignals[i] += w[i] * blame;
            }

            neurons[n].backwardsPass(blame, learningRate);
        }

        return prevErrorSignals;
    }

    public int getNumNeurons() {
        return numNeurons;
    }

    public int getInputSize() {
        return inputSize;
    }

    public Neuron[] getNeurons() {
        return neurons;
    }
}