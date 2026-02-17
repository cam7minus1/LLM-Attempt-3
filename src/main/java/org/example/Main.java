package org.example;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;

import java.io.File;
import java.io.FileReader;
import java.util.Map;

public class Main {

    public static void main(String[] args) {

        try {
            // 1. Load vocab size
            File jsonFile = new File("src/main/resources/wordEmbeddings.json");

            if (!jsonFile.exists()) {
                System.err.println("ERROR: wordEmbeddings.json not found. Run PrepData first.");
                return;
            }

            Gson gson = new Gson();
            Map<String, Object> jsonData = gson.fromJson(
                    new FileReader(jsonFile),
                    new TypeToken<Map<String, Object>>(){}.getType()
            );

            double numWordsDouble = (double) jsonData.get("numWords");
            int vocabSize = (int) numWordsDouble;

            System.out.println("Loaded vocab size: " + vocabSize);

            // 2. Network config
            int windowSize = 50;
            int perTokenSize = 4; // 3 emb + 1 pos
            int inputSize = windowSize * perTokenSize; // 200
            int hiddenSize = 32;
            int outputSize = vocabSize;
            int numHiddenLayers = 2;

            // 3. Load or create network with weights
            String weightsPath = "src/main/resources/networkWeights.json";
            Network network = Network.loadOrCreate(weightsPath, inputSize, hiddenSize, outputSize, numHiddenLayers);

            // 4. Trainer
            Trainer trainer = new Trainer(network, 0.00001f, vocabSize);

            // 5. Train on real data
            trainer.trainRealData(
                    "src/main/resources/trainingData.txt",
                    "src/main/resources/wordEmbeddings.json",
                    10
            );

            // 6. Save weights
            network.saveWeights(weightsPath);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}