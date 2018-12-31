package com.smagellan.toyml;

import org.ejml.data.DMatrixIterator;
import org.ejml.simple.SimpleMatrix;

public class MlFunctions {

    public static SimpleMatrix sigmoid(SimpleMatrix zMatrix) {
        SimpleMatrix result = zMatrix.copy();
        DMatrixIterator iter = EjmlHelpers.iterator(result);
        while (iter.hasNext()) {
            Double val = iter.next();
            iter.set(sigmoid(val));
        }
        return result;
    }

    public static double sigmoid(double z) {
        return 1.0 / (1.0 + Math.exp(-z));
    }

    public static SimpleMatrix sigmoidGradient(SimpleMatrix zMatrix) {
        SimpleMatrix result  = zMatrix.copy();
        DMatrixIterator iter = EjmlHelpers.iterator(result);
        while (iter.hasNext()) {
            Double val = iter.next();
            iter.set(sigmoidGradient(val));
        }
        return result;
    }

    public static double sigmoidGradient(double z) {
        double g = sigmoid(z);
        return g * (1 - g);
    }
}
