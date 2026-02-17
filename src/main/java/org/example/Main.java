package org.example;

//TIP To <b>Run</b> code, press <shortcut actionId="Run"/> or
// click the <icon src="AllIcons.Actions.Execute"/> icon in the gutter.
public class Main {


    static void main() {


        int size = 10;
        Network n = new Network(3, size);
        Trainer trainer = new Trainer(n, 0.1f);

        trainer.mockTrain(10, size, 100);


    }
}
