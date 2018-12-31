package com.smagellan.toyml;

import org.ejml.simple.SimpleMatrix;

public class Predictor {
    public static SimpleMatrix predict(SimpleMatrix theta1, SimpleMatrix theta2, SimpleMatrix featuresMatrixWithOnes) {
        int m = featuresMatrixWithOnes.numRows();

        SimpleMatrix h1   = MlFunctions.sigmoid(featuresMatrixWithOnes.mult(theta1.transpose()));
        SimpleMatrix h1w1 = EjmlHelpers.ones(m, 1).concatColumns(h1);
        SimpleMatrix h2   = h1w1.mult(theta2.transpose());

        return EjmlHelpers.maxPerRow(h2);
    }
}
