package org.example;

import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class Chat {

    private final Network network;
    private final Tokenizer tokenizer;
    private final int windowSize;

    public Chat(Network network, Tokenizer tokenizer, int windowSize) {
        this.network = network;
        this.tokenizer = tokenizer;
        this.windowSize = windowSize;
    }

    public void start() {
        System.out.println("Chat started. Type 'exit' to quit.");

        Scanner scanner = new Scanner(System.in);

        List<Integer> history = new ArrayList<>();

        while (true) {
            System.out.print("You: ");
            String userInput = scanner.nextLine().trim();

            if (userInput.equalsIgnoreCase("exit")) {
                System.out.println("Goodbye.");
                break;
            }

            List<Integer> userTokens = tokenizeSentence(userInput);
            history.addAll(userTokens);

            String response = generateResponse(history);

            System.out.println("Bot: " + response);

            List<Integer> botTokens = tokenizeSentence(response);
            history.addAll(botTokens);

            if (history.size() > windowSize) {
                int excess = history.size() - windowSize;
                history = history.subList(excess, history.size());
            }
        }
    }

    private List<Integer> tokenizeSentence(String sentence) {
        List<Integer> tokens = new ArrayList<>();
        String[] words = sentence.split("\\s+");

        for (String w : words) {
            int idx = tokenizer.getIndex(w);
            if (idx == -1) {
                continue;
            }
            tokens.add(idx);
        }
        return tokens;
    }

    private int sampleWithTemperature(float[] probs, float temperature) {
        float[] scaled = new float[probs.length];
        float sum = 0;
        for (int i = 0; i < probs.length; i++) {
            scaled[i] = (float) Math.exp(Math.log(probs[i] + 1e-10) / temperature);
            sum += scaled[i];
        }
        float r = (float) Math.random() * sum;
        float cumulative = 0;
        for (int i = 0; i < scaled.length; i++) {
            cumulative += scaled[i];
            if (r <= cumulative) return i;
        }
        return scaled.length - 1;
    }

    private String generateResponse(List<Integer> history) {

        List<Integer> window = getWindow(history);

        StringBuilder sb = new StringBuilder();

        for (int i = 0; i < 20; i++) {

            float[] inputVector = buildInputVector(window);

            float[] output = network.forward(inputVector);

            int nextToken = sampleWithTemperature(output, 0.8f);

            String nextWord = tokenizer.getWord(nextToken);

            if (nextWord.equals("<END>")) {
                break;
            }

            sb.append(nextWord).append(" ");

            window.remove(0);
            window.add(nextToken);
        }

        return sb.toString().trim();
    }

    private List<Integer> getWindow(List<Integer> history) {
        List<Integer> window = new ArrayList<>();

        int start = Math.max(0, history.size() - windowSize);
        for (int i = start; i < history.size(); i++) {
            window.add(history.get(i));
        }

        while (window.size() < windowSize) {
            window.add(0);
        }

        return window;
    }

    private float[] buildInputVector(List<Integer> window) {
        int perTokenSize = 129; // 128 embedding floats + 1 position float
        float[] vec = new float[windowSize * perTokenSize];

        for (int i = 0; i < windowSize; i++) {
            int token = window.get(i);
            String word = tokenizer.getWord(token);
            float[] emb = tokenizer.getEmbedding(word);

            int base = i * perTokenSize;

            if (emb != null && emb.length == 128) {
                for (int j = 0; j < 128; j++) {
                    vec[base + j] = emb[j];
                }
            }
            // else leave as zeros for padding/unknown

            // Position encoding
            vec[base + 128] = (float) i / windowSize;
        }

        return vec;
    }

    private int argmax(float[] arr) {
        int best = 0;
        float bestVal = arr[0];

        for (int i = 1; i < arr.length; i++) {
            if (arr[i] > bestVal) {
                bestVal = arr[i];
                best = i;
            }
        }
        return best;
    }
}