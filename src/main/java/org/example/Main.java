package org.example;

//TIP To <b>Run</b> code, press <shortcut actionId="Run"/> or
// click the <icon src="AllIcons.Actions.Execute"/> icon in the gutter.
public class Main {


    static void main() {

        CrossCatagoricalEntropyLoss ccel = new CrossCatagoricalEntropyLoss();

        int size = 10;
        Network n = new Network(3, size);

        float[] input = new float[128];
        for (int i = 0; i < size; i++){
            input[i] = (float)Math.random();
        }

        float[] softMaxData = n.forward(input);

        System.out.println("RESULT");
        for (int i = 0; i < softMaxData.length; i++){
            System.out.println(i+" = "+softMaxData[i]);
        }

        float[] correctResult = new float[size];
        for (int i = 0; i < size; i++){
            if (i == 0){
                correctResult[i] = 1;
            }else{
                correctResult[i] = 0;
            }
        }
        float catagoricalError = ccel.calculate(softMaxData, correctResult);
        System.out.println("Cross Catagorical Error = "+catagoricalError);
    }
}
