package org.example;

public class CrossCatagoricalEntropyLoss {

    CrossCatagoricalEntropyLoss(){}

    public float calculate(float[] softmaxData, float[] truthFlags){
        float result = 0;

        for(int i = 0; i < softmaxData.length; i++){
            double amt = (double) softmaxData[i];
            result -= (float) (truthFlags[i] * Math.log(amt));
        }

        return result;
    }

}
