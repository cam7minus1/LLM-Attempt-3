package org.example;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;

import java.io.File;
import java.io.FileReader;
import java.util.Map;

public class Main {

    public static void main(String[] args) {

        try {

            String mode = "chat";

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
            int windowSize = 10;
            int perTokenSize = 4; // 3 emb + 1 pos
            int inputSize = windowSize * perTokenSize; // 200
            int hiddenSize = 64;
            int outputSize = vocabSize;
            int numHiddenLayers = 2;

            // 3. Load or create network with weights
            String weightsPath = "src/main/resources/networkWeights.json";
            Network network = Network.loadOrCreate(weightsPath, inputSize, hiddenSize, outputSize, numHiddenLayers);

            // 4. Tokenizer
            Tokenizer tokenizer = new Tokenizer("src/main/resources/wordEmbeddings.json");

            // MODE HANDLING
            switch (mode) {

                case "train":
                    System.out.println("=== TRAIN MODE ===");
                    Trainer trainer = new Trainer(network, 0.001f, vocabSize);
                    trainer.trainRealData(
                            "src/main/resources/trainingData.txt",
                            "src/main/resources/wordEmbeddings.json",
                            100000
                    );
                    network.saveWeights(weightsPath);
                    System.out.println("Training complete.");
                    break;

                case "chat":
                    System.out.println("=== CHAT MODE ===");
                    Chat chat = new Chat(network, tokenizer, windowSize);
                    chat.start();
                    break;

                case "trainchat":
                    System.out.println("=== TRAIN + CHAT MODE ===");
                    Trainer trainer2 = new Trainer(network, 0.00001f, vocabSize);
                    trainer2.trainRealData(
                            "src/main/resources/trainingData.txt",
                            "src/main/resources/wordEmbeddings.json",
                            30000
                    );
                    network.saveWeights(weightsPath);

                    Chat chat2 = new Chat(network, tokenizer, windowSize);
                    chat2.start();
                    break;

                default:
                    System.out.println("Unknown mode: " + mode);
                    System.out.println("Usage: java -jar app.jar [train | chat | trainchat]");
            }

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}