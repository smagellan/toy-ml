package com.smagellan.toyml;

import org.apache.commons.lang3.tuple.ImmutableTriple;
import org.apache.commons.lang3.tuple.Triple;
import org.ejml.data.DMatrix;
import org.ejml.data.DMatrixRMaj;
import org.ejml.equation.Equation;
import org.ejml.simple.SimpleMatrix;
import org.slf4j.LoggerFactory;
import us.hebi.matlab.mat.ejml.Mat5Ejml;
import us.hebi.matlab.mat.format.Mat5;
import us.hebi.matlab.mat.format.Mat5File;
import us.hebi.matlab.mat.types.Matrix;
import us.hebi.matlab.mat.types.Source;
import us.hebi.matlab.mat.types.Sources;

import java.io.File;
import java.io.IOException;

public class Main {
    private static final org.slf4j.Logger logger = LoggerFactory.getLogger(Main.class);

    public static final File EX4_DATA1 = new File("/home/vladimir/projects/coursera/machine-learning/machine-learning-ex4/ex4/ex4data1.mat");
    public static final File EX4_WEIGHTS = new File("/home/vladimir/projects/coursera/machine-learning/machine-learning-ex4/ex4/ex4weights.mat");


    public void doFit(Triple<DMatrix, DMatrix, Long> params) {
        doFit(params.getLeft(), params.getMiddle(), params.getRight());
    }

    public void doFit(DMatrix featuresMatrix, DMatrix yMatrix, long seed) {
        int inputLayerSize  = 400;
        int hiddenLayerSize = 25;
        int numLabels       = 10;
        double lambda       = 1;

        SimpleMatrix theta1 = EjmlHelpers.randInitializeWeights(inputLayerSize, hiddenLayerSize, seed);
        SimpleMatrix theta2 = EjmlHelpers.randInitializeWeights(hiddenLayerSize, numLabels, 2 * seed);


        Equation nnParamsUnroller = new Equation();
        nnParamsUnroller.alias(theta1, "Theta1");
        nnParamsUnroller.alias(theta2, "Theta2");
        SimpleMatrix nnParams = new SimpleMatrix(theta1.getNumElements() + theta2.getNumElements(), 1);
        nnParamsUnroller.alias(nnParams, "nn_params");
        nnParamsUnroller.process("nn_params = [Theta1(:), Theta2(:)]");
        nnParams = nnParams.transpose();


        Equation featuresMatrixEnricher    = new Equation();
        SimpleMatrix featuresMatrixWithOnes = new SimpleMatrix(featuresMatrix.getNumRows(), featuresMatrix.getNumCols() + 1);
        featuresMatrixEnricher.alias(featuresMatrix, "X");
        featuresMatrixEnricher.alias(featuresMatrixWithOnes, "Xw1");
        featuresMatrixEnricher.alias(featuresMatrix.getNumRows(), "m");
        featuresMatrixEnricher.process("Xw1 = [ones(m, 1), X]");

        CostFunction cf = new CostFunction(inputLayerSize, hiddenLayerSize, numLabels,
                featuresMatrixWithOnes, createYVec(yMatrix, 10), lambda);
        cf.compute(nnParams);

        OptimizationResult optimizationResult = OptimizationFunctions.fmincg(cf, nnParams, 50);
        SimpleMatrix learnedTheta1 = optimizationResult.getxVal().rows(0, theta1.getNumElements());
        learnedTheta1.reshape(hiddenLayerSize, inputLayerSize + 1);

        SimpleMatrix learnedTheta2 = optimizationResult.getxVal().rows(theta1.getNumElements(), theta1.getNumElements() + theta2.getNumElements());
        learnedTheta2.reshape(numLabels, hiddenLayerSize + 1);

        SimpleMatrix predicted = Predictor.predict(learnedTheta1, learnedTheta2, featuresMatrixWithOnes);
        int predictedCount = predictionsMatchedCount(predicted, SimpleMatrix.wrap(yMatrix));
        logger.debug("accuracy: {}", ((double)predictedCount) * 100 / yMatrix.getNumRows());
    }

    public static Triple<DMatrix, DMatrix, Long> loadParams(long seed) throws IOException {
        DMatrix featuresMatrix;
        DMatrix yMatrix;
        try(Source source  = Sources.openFile(EX4_DATA1); Mat5File file = Mat5.newReader(source).readMat()) {
            featuresMatrix = loadMatrix(file, "X");
            yMatrix        = loadMatrix(file, "y");
        }
        return ImmutableTriple.of(featuresMatrix, yMatrix, seed);
    }

    public static void main(String[] args) throws IOException {
        Triple<DMatrix, DMatrix, Long> params = loadParams(100);
        new Main().doFit(params);
    }

    public static int predictionsMatchedCount(SimpleMatrix predicted, SimpleMatrix yMatrix) {
        int matchCount = 0;
        for (int rowIdx = 0; rowIdx < yMatrix.numRows(); ++rowIdx) {
            //Matlab dataset adjusted by one to simplify indexing since matlab indexes start from 1
            //dont forget to subtract it here
            int expectedVal = (int)yMatrix.get(rowIdx, 0) - 1;
            int predictedVal = (int)predicted.get(rowIdx, 0);
            if (expectedVal == predictedVal) {
                ++matchCount;
            }
        }
        return matchCount;
    }

    public static SimpleMatrix createYVec(DMatrix yMatrix, int numClasses) {
        DMatrixRMaj result = new DMatrixRMaj(yMatrix.getNumRows(), numClasses);
        for (int rowIdx = 0; rowIdx < yMatrix.getNumRows(); ++rowIdx) {
            int exampleKlass = (int)yMatrix.get(rowIdx, 0);
            //Matlab dataset adjusted by one to simplify indexing since matlab indexes start from 1
            //dont forget to subtract it here
            result.set(rowIdx, exampleKlass - 1, 1);
        }
        return SimpleMatrix.wrap(result);
    }

    public static DMatrixRMaj loadMatrix(Mat5File file, String name) {
        Matrix matrix = file.getMatrix(name);
        return Mat5Ejml.convert(matrix, new DMatrixRMaj(matrix.getNumRows(), matrix.getNumCols()));
    }
}
