package com.smagellan.toyml;

import org.ejml.data.DMatrixIterator;
import org.ejml.simple.SimpleMatrix;

import java.util.Random;

public class EjmlHelpers {
    public static final double EPSILON_INIT = 0.12;

    public static DMatrixIterator iterator(SimpleMatrix src) {
        return iterator(src, true);
    }

    public static DMatrixIterator iterator(SimpleMatrix src, boolean rowMajor) {
        return src.iterator(rowMajor, 0, 0, src.numRows() - 1, src.numCols() - 1);
    }

    public static SimpleMatrix ones(int numRows, int numCols) {
        SimpleMatrix result = new SimpleMatrix(numRows, numCols);
        return result.plus(1);
    }

    public static SimpleMatrix zeros(int numRows, int numCols) {
        return new SimpleMatrix(numRows, numCols);
    }

    public static SimpleMatrix mul(SimpleMatrix src, double m) {
        SimpleMatrix result  = src.copy();
        DMatrixIterator iter = iterator(result);
        while (iter.hasNext()) {
            Double val = iter.next();
            iter.set(val * m);
        }
        return result;
    }

    public static double matrixAsScalar(SimpleMatrix src) {
        if (src.getNumElements() != 1) {
            throw new IllegalArgumentException("matrix has " + src.getNumElements() + "; expect 1");
        }
        return src.get(0, 0);
    }

    public static SimpleMatrix singleton(double src) {
        SimpleMatrix result = new SimpleMatrix(1, 1);
        result.set(0, 0, src);
        return result;
    }

    public static SimpleMatrix randInitializeWeights(int lIn, int lOut) {
        SimpleMatrix w = EjmlHelpers.zeros(lOut, lIn + 1);
        Random r = new Random();
        DMatrixIterator iter = EjmlHelpers.iterator(w);
        while (iter.hasNext()) {
            iter.next();
            double val = r.nextDouble() * 2 * EPSILON_INIT - EPSILON_INIT;
            iter.set(val);
        }
        return w;
    }
}
