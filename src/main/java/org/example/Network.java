package org.example;

public class Network {
    Layer[] layers;

    // Current configurations makes 3 layers 128 size = 49,465 param model
    public Network(int numLayers, int size){

        layers = new Layer[numLayers];

        for(int i = 0; i < numLayers; i++){
            boolean isOutputLayer = false;

            if(i == numLayers-1){
                isOutputLayer = true;
            }

            layers[i] = new Layer(size, size, isOutputLayer);
        }
    }

    public float[] softMaxActivation(float[] input){
        double eKDenom = 0;
        float[] result = new float[input.length];
        for (int i = 0; i < input.length; i++){
            double k = (double) input[i];
            eKDenom += Math.exp(k);
        }

        for (int i = 0; i < input.length; i++){
            double res = Math.exp(input[i])/eKDenom;
            result[i] =  (float) res;
        }

        return result;
    }

    public float[] forward(float[] input){

        float[] previous = input;

        for(int i = 0; i < layers.length; i++){
            previous = layers[i].forwardPass(previous);
        }

        // Return the soft max activation function of the last layer
        return softMaxActivation(previous);
    }

    public void backwards(float[] blames, float learningRate){
        float[] prevErrorSignal = blames;
        // Start at the last layer and work backwards
        for (int i = layers.length -1; i >= 0; i--){
            prevErrorSignal = layers[i].backwardsPass(prevErrorSignal, learningRate);
        }
    }

}
