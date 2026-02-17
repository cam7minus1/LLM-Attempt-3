package org.example;

public class Neuron {

    float[] weights;
    float bias;
    boolean isOutputLayer;
    float lastActivation;

    public Neuron(int numWeights, boolean isOutputLayer){
        this.weights = new float[numWeights];
        this.bias = (float) Math.random();
        this.isOutputLayer = isOutputLayer;
        this.lastActivation = -1.0f;

        for (int i = 0; i < numWeights; i++){
            weights[i] = (float) Math.random();
        }
    }

    public void setBias(float newBias){
        this.bias = newBias;
        System.out.println("New bias is "+newBias);
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

    public float Relu(float input){
        if (input > 0){
            return input;
        }else{
            return 0.0f;
        }
    }

    public float forwardPass(float[] inputs){
        float result = 0.0f;

        // Weighted sum + Bias
        for (int i = 0; i < weights.length; i++){
            result += weights[i] * inputs[i];
        }

        // Add the bias;
        result += bias;

        if (isOutputLayer){
            lastActivation = result;
            return lastActivation;
        }else{
            lastActivation = Relu(result);
            return  lastActivation;
        }

    }

    @Override
    public String toString(){
        String output = "Node Data \n Bias: "+this.bias;

        for (int i = 0; i < weights.length; i++){
            output += "\n Weight "+i+" = "+weights[i];
        }

        return  output;
    }

}
