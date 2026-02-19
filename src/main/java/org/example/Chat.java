package org.example;

import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class Chat {

    private final Network network;
    private final Tokenizer tokenizer;
    private final int windowSize;
    private final int embedSize;

    public Chat(Network network, Tokenizer tokenizer, int windowSize, int embedSize) {
        this.network = network;
        this.tokenizer = tokenizer;
        this.windowSize = windowSize;
        this.embedSize = embedSize;
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
                history = history.subList(history.size() - windowSize, history.size());
            }
        }
    }

    private List<Integer> tokenizeSentence(String sentence) {
        List<Integer> tokens = new ArrayList<>();
        for (String w : sentence.split("\\s+")) {
            int idx = tokenizer.getIndex(w);
            if (idx != -1) tokens.add(idx);
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
            float[][] tokenWindow = buildTokenWindow(window);
            float[] output = network.forward(tokenWindow);
            int nextToken = sampleWithTemperature(output, 0.8f);
            String nextWord = tokenizer.getWord(nextToken);

            if (nextWord.equals("<END>")) break;

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
            window.add(0, 0);
        }
        return window;
    }

    private float[][] buildTokenWindow(List<Integer> window) {
        float[][] tokens = new float[windowSize][];
        for (int i = 0; i < windowSize; i++) {
            String word = tokenizer.getWord(window.get(i));
            float[] emb = tokenizer.getEmbedding(word);
            if (emb != null && emb.length == embedSize) {
                tokens[i] = emb;
            } else {
                tokens[i] = new float[embedSize];
            }
        }
        return tokens;
    }
}