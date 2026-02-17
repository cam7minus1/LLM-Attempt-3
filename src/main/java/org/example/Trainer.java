package org.example;

public class Trainer {

    Network network;
    CrossCatagoricalEntropyLoss ccel;
    float learningRate;

    public Trainer(Network network, float learningRate){
        this.network = network;
        this.ccel = new CrossCatagoricalEntropyLoss();
        this.learningRate = learningRate;
    }

    public float[][] getFakeTrainingData(int numElements, int sizeOfData){
        float[][] data = new float[numElements][sizeOfData];
        java.util.Random rand = new java.util.Random();

        for (int i = 0; i < numElements; i++) {
            for (int k = 0; k < sizeOfData; k++) {
                data[i][k] = rand.nextFloat();
            }
        }

        return data;
    }

    public float[][] getFakeLabels(int numElements, int numOutputNodes) {
        float[][] labels = new float[numElements][numOutputNodes];
        java.util.Random rand = new java.util.Random();

        for (int i = 0; i < numElements; i++) {
            int index = rand.nextInt(numOutputNodes); // choose which node is "correct"
            labels[i][index] = 1.0f;                  // one-hot encoding
        }

        return labels;
    }

    public void train(float[][] trainingData, float[][] labels, int epochs) {
        System.out.println("Beginning training...");

        for (int e = 0; e < epochs; e++) {

            float epochLoss = 0f;

            for (int i = 0; i < trainingData.length; i++) {

                float[] inputData = trainingData[i];
                float[] expectedOutput = labels[i];

                // Forward pass
                float[] softmaxOutput = this.network.forward(inputData);

                // Loss
                float loss = ccel.calculate(softmaxOutput, expectedOutput);
                epochLoss += loss;

                // Output blame (softmax - one-hot)
                float[] outputBlame = new float[softmaxOutput.length];
                for (int j = 0; j < softmaxOutput.length; j++) {
                    outputBlame[j] = softmaxOutput[j] - expectedOutput[j];
                }

                // Backprop
                this.network.backwards(outputBlame, this.learningRate);
            }

            // Average loss for this epoch
            float avgLoss = epochLoss / trainingData.length;

            System.out.println("Epoch " + e + " - Avg Loss: " + avgLoss);
        }
    }

    public void mockTrain(int numElementsToTrain, int dataSize, int epocs){
        float[][] trainingData = getFakeTrainingData(numElementsToTrain, dataSize);
        float[][] labels = getFakeLabels(numElementsToTrain, dataSize);
        train(trainingData, labels, epocs);
    }

}
