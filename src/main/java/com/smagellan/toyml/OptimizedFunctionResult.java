package com.smagellan.toyml;

import org.ejml.simple.SimpleMatrix;

public class OptimizedFunctionResult {
    private final double fnValue;
    private final SimpleMatrix gradientValue;

    public OptimizedFunctionResult(double fnValue, SimpleMatrix gradientValue) {
        this.fnValue = fnValue;
        this.gradientValue = gradientValue;
    }

    public double getFnValue() {
        return fnValue;
    }

    public SimpleMatrix getGradientValue() {
        return gradientValue;
    }
}
