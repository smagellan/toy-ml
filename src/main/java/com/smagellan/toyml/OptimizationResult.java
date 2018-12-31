package com.smagellan.toyml;

import org.ejml.simple.SimpleMatrix;

public class OptimizationResult {
    private final SimpleMatrix xVal;
    private final SimpleMatrix fXVal;
    private final int numIterations;


    public OptimizationResult(SimpleMatrix xVal, SimpleMatrix fXVal, int numIterations) {
        this.xVal = xVal;
        this.fXVal = fXVal;
        this.numIterations = numIterations;
    }

    public SimpleMatrix getxVal() {
        return xVal;
    }

    public SimpleMatrix getfXVal() {
        return fXVal;
    }

    public int getNumIterations() {
        return numIterations;
    }
}
