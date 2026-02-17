package org.example;

public class CrossCatagoricalEntropyLoss {

    CrossCatagoricalEntropyLoss(){}

    public float calculate(float[] softmaxData, float[] truthFlags){
        float result = 0f;
        final float eps = 1e-8f;

        for (int i = 0; i < softmaxData.length; i++) {
            float p = Math.max(softmaxData[i], eps);
            result -= truthFlags[i] * (float)Math.log(p);
        }

        return result;
    }

}
