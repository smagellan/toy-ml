package com.smagellan.toyml;

import org.ejml.simple.SimpleMatrix;
import org.slf4j.LoggerFactory;

import java.util.function.Function;
import java.util.stream.DoubleStream;

public class CostFunction implements Function<SimpleMatrix, OptimizedFunctionResult> {
    private static final org.slf4j.Logger logger = LoggerFactory.getLogger(CostFunction.class);

    private final int inputLayerSize;
    private final int hiddenLayerSize;
    private final int numLabels;
    private final SimpleMatrix featuresMatrixWithOnes;
    private final SimpleMatrix yVec;
    private final double lambda;

    public CostFunction(int inputLayerSize, int hiddenLayerSize, int numLabels, SimpleMatrix featuresMatrixWithOnes, SimpleMatrix yVec, double lambda) {
        this.inputLayerSize = inputLayerSize;
        this.hiddenLayerSize = hiddenLayerSize;
        this.numLabels = numLabels;
        this.featuresMatrixWithOnes = featuresMatrixWithOnes;
        this.yVec = yVec;
        this.lambda = lambda;
    }

    public OptimizedFunctionResult compute(SimpleMatrix nnParams) {
        int theta1NumElements = hiddenLayerSize * (inputLayerSize + 1);
        int theta2NumElements = numLabels * (hiddenLayerSize + 1);

        SimpleMatrix theta1 = nnParams.rows(0, theta1NumElements);
        theta1.reshape(hiddenLayerSize, inputLayerSize + 1);
        SimpleMatrix theta2 = nnParams.rows(theta1NumElements, theta1NumElements + theta2NumElements);
        theta2.reshape(numLabels, hiddenLayerSize + 1);

        int trainExamplesCount = featuresMatrixWithOnes.numRows();

        SimpleMatrix z2     = featuresMatrixWithOnes.mult(theta1.transpose());
        SimpleMatrix act2   = MlFunctions.sigmoid(z2);
        SimpleMatrix act2w1 = EjmlHelpers.ones(trainExamplesCount, 1).concatColumns(act2);
        SimpleMatrix z3     = act2w1.mult(theta2.transpose());
        SimpleMatrix act3   = MlFunctions.sigmoid(z3);

        SimpleMatrix tmp    = yVec.negative().plus(1).elementMult(act3.negative().plus(1).elementLog());
        double jValue = yVec.negative().elementMult(act3.elementLog()).minus(tmp).elementSum() / trainExamplesCount;
        //logger.debug("non-regularized J: {}", jValue);

        SimpleMatrix theta1C = theta1.copy();
        theta1C.setColumn(0, 0, DoubleStream.generate(() -> 0).limit(theta1C.numRows()).toArray());

        SimpleMatrix theta2C = theta2.copy();
        theta2C.setColumn(0, 0, DoubleStream.generate(() -> 0).limit(theta2C.numRows()).toArray());

        double jValueReg = (theta1C.elementPower(2).elementSum() + theta2C.elementPower(2).elementSum()) * lambda / (2 * trainExamplesCount);

        jValue += jValueReg;
        //logger.debug("regularized J: {}", jValue);

        SimpleMatrix sigma3   = act3.minus(yVec);
        SimpleMatrix sigma2P1 = theta2.transpose().mult(sigma3.transpose());
        sigma2P1              = sigma2P1.rows(1, sigma2P1.numRows());
        SimpleMatrix sigma2   = sigma2P1.transpose().elementMult(MlFunctions.sigmoidGradient(z2));

        SimpleMatrix delta2   = sigma3.transpose().mult(act2w1);
        SimpleMatrix delta1   = sigma2.transpose().mult(featuresMatrixWithOnes);

        SimpleMatrix theta2Grad = delta2.plus(EjmlHelpers.mul(theta2C, lambda)).divide(trainExamplesCount);
        SimpleMatrix theta1Grad = delta1.plus(EjmlHelpers.mul(theta1C, lambda)).divide(trainExamplesCount);

        theta1Grad.reshape(1, theta1Grad.getNumElements());
        theta2Grad.reshape(1, theta2Grad.getNumElements());
        SimpleMatrix unrolledGrad = theta1Grad.concatColumns(theta2Grad);
        //logger.debug("grad sum: {}", unrolledGrad.elementSum());

        return new OptimizedFunctionResult(jValue, unrolledGrad.transpose());
    }

    @Override
    public OptimizedFunctionResult apply(SimpleMatrix simpleMatrix) {
        return compute(simpleMatrix);
    }
}
