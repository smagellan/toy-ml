package com.smagellan.toyml;

import org.ejml.simple.SimpleMatrix;

public class Predictor {
    public static SimpleMatrix predict(SimpleMatrix theta1, SimpleMatrix theta2, SimpleMatrix featuresMatrixWithOnes) {
        int m = featuresMatrixWithOnes.numRows();

        SimpleMatrix h1   = MlFunctions.sigmoid(featuresMatrixWithOnes.mult(theta1.transpose()));
        SimpleMatrix h1w1 = EjmlHelpers.ones(m, 1).concatColumns(h1);
        SimpleMatrix h2   = h1w1.mult(theta2.transpose());


        SimpleMatrix result = new SimpleMatrix(m, 1);
        for (int rowNum = 0; rowNum < h2.numRows(); ++rowNum) {
            double max = h2.get(rowNum, 0);
            int index = 0;
            for (int colNum = 0; colNum < h2.numCols(); ++colNum) {
                double elVal = h2.get(rowNum, colNum);
                if (elVal > max) {
                    max = elVal;
                    index = colNum;
                }
            }
            result.set(rowNum, 0, index);
        }

        return result;
    }
}
