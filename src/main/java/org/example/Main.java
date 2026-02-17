package org.example;

//TIP To <b>Run</b> code, press <shortcut actionId="Run"/> or
// click the <icon src="AllIcons.Actions.Execute"/> icon in the gutter.
public class Main {


    static void main() {


        int size = 100;
        Network n = new Network(5, size);
        Trainer trainer = new Trainer(n, 0.001f);

        trainer.mockTrain(100, size, 100);


    }
}
