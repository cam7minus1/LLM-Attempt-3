package org.example;

import java.io.*;
import java.util.*;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.reflect.TypeToken;

public class PrepData {

    public static void main(String[] args) {
        System.out.println("Preparing training data...");
        PrepData();
        System.out.println("Done.");
    }

    public static void PrepData() {

        // -------------------------------
        // 1. Load raw training data
        // -------------------------------
        List<String> cleanedLines = new ArrayList<>();

        File trainingFile = new File("src/main/resources/trainingData.txt");

        if (!trainingFile.exists()) {
            System.err.println("ERROR: trainingData.txt not found in src/main/resources/");
            return;
        }

        try (BufferedReader br = new BufferedReader(new FileReader(trainingFile))) {
            String line;

            while ((line = br.readLine()) != null) {

                // Strip prefixes
                if (line.startsWith("Human 1: ")) {
                    line = line.substring("Human 1: ".length());
                } else if (line.startsWith("Human 2: ")) {
                    line = line.substring("Human 2: ".length());
                }

                cleanedLines.add(line);
            }

        } catch (Exception e) {
            e.printStackTrace();
            return;
        }

        // -------------------------------
        // 2. Overwrite the SAME trainingData.txt with cleaned lines
        // -------------------------------
        try (FileWriter fw = new FileWriter(trainingFile, false)) {
            for (String line : cleanedLines) {
                fw.write(line + "\n");
            }
        } catch (Exception e) {
            e.printStackTrace();
        }

        System.out.println("Cleaned training data written back to: " + trainingFile.getAbsolutePath());

        // -------------------------------
        // 3. Build vocabulary from cleaned data
        // -------------------------------
        Set<String> vocab = new HashSet<>();

        for (String line : cleanedLines) {
            String[] words = line.toLowerCase().split("[^a-zA-Z0-9']+");
            for (String w : words) {
                if (!w.isEmpty()) {
                    vocab.add(w);
                }
            }
        }

        // -------------------------------
        // 4. Load existing embeddings JSON (if exists)
        // -------------------------------
        File outFile = new File("src/main/resources/wordEmbeddings.json");

        Map<String, Object> finalJson = new LinkedHashMap<>();
        Map<String, float[]> embeddings = new HashMap<>();
        Set<String> usedVectors = new HashSet<>();

        Gson gson = new GsonBuilder().setPrettyPrinting().create();

        if (outFile.exists()) {
            try (FileReader fr = new FileReader(outFile)) {

                finalJson = gson.fromJson(fr, new TypeToken<Map<String, Object>>(){}.getType());

                Map<String, List<Double>> loaded = (Map<String, List<Double>>) finalJson.get("embeddings");

                for (String word : loaded.keySet()) {
                    List<Double> vecD = loaded.get(word);
                    float[] vec = new float[]{
                            vecD.get(0).floatValue(),
                            vecD.get(1).floatValue(),
                            vecD.get(2).floatValue()
                    };
                    embeddings.put(word, vec);

                    usedVectors.add(vec[0] + "," + vec[1] + "," + vec[2]);
                }

            } catch (Exception e) {
                e.printStackTrace();
                return;
            }
        }

        // -------------------------------
        // 5. Add ONLY new words
        // -------------------------------
        Random rand = new Random();

        for (String word : vocab) {
            if (embeddings.containsKey(word)) {
                continue; // already exists
            }

            float[] vec;
            String signature;

            do {
                vec = new float[3];
                vec[0] = rand.nextFloat();
                vec[1] = rand.nextFloat();
                vec[2] = rand.nextFloat();

                signature = vec[0] + "," + vec[1] + "," + vec[2];

            } while (usedVectors.contains(signature));

            usedVectors.add(signature);
            embeddings.put(word, vec);
        }

        // -------------------------------
        // 6. Save updated JSON
        // -------------------------------
        finalJson = new LinkedHashMap<>();
        finalJson.put("numWords", embeddings.size());
        finalJson.put("embeddings", embeddings);

        try (FileWriter fw = new FileWriter(outFile)) {
            gson.toJson(finalJson, fw);
            System.out.println("Updated embeddings. Total words: " + embeddings.size());
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}