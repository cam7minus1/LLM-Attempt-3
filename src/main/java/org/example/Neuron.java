package org.example;

import java.util.Random;

public class Neuron {

    boolean isOutputLayer;
    float bias;
    float lastActivation;
    float[] weights;
    float[] lastInputs;

    public Neuron(int numWeights, boolean isOutputLayer){
        this.weights = new float[numWeights];
        this.isOutputLayer = isOutputLayer;
        this.lastActivation = -1.0f;
        this.lastInputs = new float[numWeights];

        Random rand = new Random();
        float scale = (float) Math.sqrt(2.0 / numWeights);

        for (int i = 0; i < numWeights; i++) {
            weights[i] = (float) rand.nextGaussian() * scale;
        }

        this.bias = 0.0f;
    }

    public void setBias(float newBias){
        this.bias = newBias;
    }

    public float getBias(){
        return this.bias;
    }

    public void setWeights(float[] newWeights){
        for (int i = 0; i < weights.length; i++){
            weights[i] = newWeights[i];
        }
    }

    public float[] getWeights(){
        return this.weights;
    }

    public float Relu(float input) {
        if (input > 0) {
            return input;
        } else {
            return 0.01f * input;
        }
    }

    public float forwardPass(float[] inputs){
        this.lastInputs = inputs.clone();

        float result = 0.0f;

        for (int i = 0; i < weights.length; i++){
            result += weights[i] * inputs[i];
        }

        result += bias;

        if (isOutputLayer){
            lastActivation = result;
            return lastActivation;
        }else{
            lastActivation = Relu(result);
            return  lastActivation;
        }
    }

    public void backwardsPass(float blameSignal, float learningRate){
        this.bias = this.bias - (learningRate * blameSignal);

        for (int i = 0; i < weights.length; i++){
            weights[i] = weights[i] - (learningRate * blameSignal * lastInputs[i]);
        }
    }
}