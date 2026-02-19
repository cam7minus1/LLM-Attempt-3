package org.example;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

import java.io.*;

public class Config {

    public int embedSize;
    public int numHeads;
    public int numBlocks;
    public int ffSize;
    public int windowSize;
    public float learningRate;
    public int epochs;
    public int batchSize;
    public int vocabSize;

    private static final String CONFIG_PATH = "src/main/resources/config.json";

    public static Config load() {
        try {
            Gson gson = new Gson();
            return gson.fromJson(new FileReader(CONFIG_PATH), Config.class);
        } catch (Exception e) {
            System.out.println("Could not load config.json, using defaults.");
            return defaults();
        }
    }

    public void save() {
        try {
            Gson gson = new GsonBuilder().setPrettyPrinting().create();
            try (FileWriter fw = new FileWriter(CONFIG_PATH)) {
                gson.toJson(this, fw);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static Config defaults() {
        Config c = new Config();
        c.embedSize = 256;
        c.numHeads = 8;
        c.numBlocks = 8;
        c.ffSize = 1024;
        c.windowSize = 20;
        c.learningRate = 0.0001f;
        c.epochs = 50000;
        c.batchSize = 32;
        c.vocabSize = 0;
        return c;
    }
}