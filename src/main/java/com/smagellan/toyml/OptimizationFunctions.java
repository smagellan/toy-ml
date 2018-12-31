package com.smagellan.toyml;

import org.apache.commons.lang3.time.StopWatch;
import org.ejml.simple.SimpleMatrix;
import org.slf4j.LoggerFactory;

import static com.smagellan.toyml.EjmlHelpers.matrixAsScalar;

import java.util.concurrent.TimeUnit;
import java.util.function.Function;

public class OptimizationFunctions {
    private static final org.slf4j.Logger logger = LoggerFactory.getLogger(OptimizationFunctions.class);

    public static final double RHO = 0.01;
    public static final double SIG = 0.5;
    public static final double INT = 0.1;
    public static final int EXT = 3;
    public static final int MAX = 20;
    public static final int RATIO = 100;
    public static final double REALMIN = Double.MIN_NORMAL;


    public static OptimizationResult fmincg(Function<SimpleMatrix, OptimizedFunctionResult> fn, SimpleMatrix fnParams, int length) {
        return fmincg(fn, fnParams, length, 1);
    }


    /*
    % Minimize a continuous differentialble multivariate function. Starting point
    % Minimize a continuous differentialble multivariate function. Starting point
    % is given by "X" (D by 1), and the function named in the string "f", must
    % return a function value and a vector of partial derivatives. The Polack-
    % Ribiere flavour of conjugate gradients is used to compute search directions,
    % and a line search using quadratic and cubic polynomial approximations and the
    % Wolfe-Powell stopping criteria is used together with the slope ratio method
    % for guessing initial step sizes. Additionally a bunch of checks are made to
    % make sure that exploration is taking place and that extrapolation will not
    % be unboundedly large. The "length" gives the length of the run: if it is
    % positive, it gives the maximum number of line searches, if negative its
    % absolute gives the maximum allowed number of function evaluations. You can
    % (optionally) give "length" a second component, which will indicate the
    % reduction in function value to be expected in the first line-search (defaults
    % to 1.0). The function returns when either its length is up, or if no further
    % progress can be made (ie, we are at a minimum, or so close that due to
    % numerical problems, we cannot get any closer). If the function terminates
    % within a few iterations, it could be an indication that the function value
    % and derivatives are not consistent (ie, there may be a bug in the
    % implementation of your "f" function). The function returns the found
    % solution "X", a vector of function values "fX" indicating the progress made
    % and "i" the number of iterations (line searches or function evaluations,
    % depending on the sign of "length") used.
    %
    % Usage: [X, fX, i] = fmincg(f, X, options, P1, P2, P3, P4, P5)
    %
    % See also: checkgrad
    %
    % Copyright (C) 2001 and 2002 by Carl Edward Rasmussen. Date 2002-02-13
    %
    %
    % (C) Copyright 1999, 2000 & 2001, Carl Edward Rasmussen
    %
    % Permission is granted for anyone to copy, use, or modify these
    % programs and accompanying documents for purposes of research or
    % education, provided this copyright notice is retained, and note is
    % made of any changes that have been made.
    %
    % These programs and documents are distributed without any warranty,
    % express or implied.  As the programs were written for research
    % purposes only, they have not been tested to the degree that would be
    % advisable in any important application.  All use of these programs is
    % entirely at the user's own risk.
    %
    * */
    public static OptimizationResult fmincg(Function<SimpleMatrix, OptimizedFunctionResult> fn, SimpleMatrix fnParams, int length, int red) {
        StopWatch sw = StopWatch.createStarted();
        boolean lsFailed = false;
        int i = 0;
        SimpleMatrix X = fnParams.copy();

        OptimizedFunctionResult fnResult = fn.apply(X);

        i += length < 0 ? 1 : 0;
        double f1 = fnResult.getFnValue();
        SimpleMatrix df1 = fnResult.getGradientValue();
        SimpleMatrix s = df1.negative();
        double d1 = matrixAsScalar(s.negative().transpose().mult(s));
        double z1 = red / (1 - d1);


        SimpleMatrix fX = new SimpleMatrix(0, 0);
        while (i < Math.abs(length)) {
            i += length > 0 ? 1 : 0;
            SimpleMatrix X0 = X.copy();
            double f0 = f1;
            SimpleMatrix df0 = df1.copy();
            X = X.plus(EjmlHelpers.mul(s, z1));

            OptimizedFunctionResult fnResult2 = fn.apply(X);
            double f2 = fnResult2.getFnValue();
            SimpleMatrix df2 = fnResult2.getGradientValue();
            i += (length < 0) ? 1 : 0;
            double d2 = matrixAsScalar(df2.transpose().mult(s));

            double f3 = f1;
            double d3 = d1;
            double z3 = -z1;
            int m = (length > 0) ? MAX : Math.min(MAX, -length - i);
            boolean success = false;
            double limit = -1;
            while (true) {
                while (((f2 > f1 + z1 * RHO * d1) || (d2 > -SIG * d1)) && (m > 0)) {
                    limit = z1;
                    double z2;
                    if (f2 > f1) {
                        z2 = z3 - (0.5 * d3 * z3 * z3) / (d3 * z3 + f2 - f3);
                    } else {
                        double A = 6 * (f2 - f3) / z3 + 3 * (d2 + d3);
                        double B = 3 * (f3 - f2) - z3 * (d3 + 2 * d2);
                        z2 = (Math.sqrt(B * B - A * d2 * z3 * z3) - B) / A;
                    }
                    if (Double.isNaN(z2) || Double.isInfinite(z2)) {
                        z2 = z3 / 2;
                    }

                    z2 = Math.max(Math.min(z2, INT * z3), (1 - INT) * z3);
                    z1 = z1 + z2;
                    X = X.plus(EjmlHelpers.mul(s, z2));

                    fnResult2 = fn.apply(X);
                    f2 = fnResult2.getFnValue();
                    df2 = fnResult2.getGradientValue();

                    --m;
                    i += (length < 0) ? 1 : 0;
                    d2 = matrixAsScalar(df2.transpose().mult(s));
                    z3 = z3 - z2;
                }

                if (f2 > f1 + z1 * RHO * d1 || d2 > -SIG * d1) {
                    break;
                } else {
                    if (d2 > SIG * d1) {
                        success = true;
                        break;
                    } else {
                        if (m == 0) {
                            break;
                        }
                    }
                }

                double A = 6 * (f2 - f3) / z3 + 3 * (d2 + d3);
                double B = 3 * (f3 - f2) - z3 * (d3 + 2 * d2);
                double z2 = -d2 * z3 * z3 / (B + Math.sqrt(B * B - A * d2 * z3 * z3));

                //no isreal here. Math.sqrt may produce Exception?
                if (Double.isNaN(z2) || Double.isInfinite(z2) || z2 < 0) {
                    z2 = (limit < -0.5) ?
                            z1 * (EXT - 1) :
                            (limit - z1) / 2;
                } else {
                    //extraplation beyond max?
                    if ((limit > -0.5) && (z2 + z1 > limit)) {
                        z2 = (limit - z1) / 2;
                    } else {
                        //extrapolation beyond limit
                        if ((limit < -0.5) && (z2 + z1 > z1 * EXT)) {
                            z2 = z1 * (EXT - 1.0);
                        } else {
                            if (z2 < -z3 * INT) {
                                z2 = -z3 * INT;
                            } else {
                                // too close to limit?
                                if ((limit > -0.5) && (z2 < (limit - z1) * (1.0 - INT))) {
                                    z2 = (limit - z1) * (1.0 - INT);
                                }
                            }
                        }
                    }
                }
                f3 = f2;
                d3 = d2;
                z3 = -z2;
                z1 = z1 + z2;
                X = X.plus(EjmlHelpers.mul(s, z2));

                fnResult2 = fn.apply(X);
                f2 = fnResult2.getFnValue();
                df2 = fnResult2.getGradientValue();

                --m;
                i += (length < 0) ? 1 : 0;
                d2 = matrixAsScalar(df2.transpose().mult(s));
            }

            if (success) {
                f1 = f2;
                fX = fX.concatColumns(EjmlHelpers.singleton(f1));
                logger.debug("Iteration {} | Cost: {}", i, f1);
                double divisor = matrixAsScalar(df1.transpose().mult(df1));
                double numerator = matrixAsScalar(df2.transpose().mult(df2).minus(df1.transpose().mult(df2)));
                s = EjmlHelpers.mul(s, numerator / divisor).minus(df2);
                SimpleMatrix tmp = df1;
                df1 = df2;
                df2 = tmp;
                d2 = matrixAsScalar(df1.transpose().mult(s));
                if (d2 > 0) {
                    s = df1.negative();
                    d2 = matrixAsScalar(s.negative().transpose().mult(s));
                }
                z1 = z1 * Math.min(RATIO, d1 / (d2 - REALMIN));
                d1 = d2;
                lsFailed = false;
            } else {
                X = X0;
                f1 = f0;
                df1 = df0; //restore point from before failed line search
                if (lsFailed || i > Math.abs(length)) {
                    break;
                }
                SimpleMatrix tmp = df1;
                df1 = df2;
                df2 = tmp;
                s = df1.negative();
                d1 = matrixAsScalar(s.negative().transpose().mult(s));
                z1 = 1 / (1 - d1);
                lsFailed = true;
            }
        }
        logger.debug("optimization took {} millis and {} iterations", sw.getTime(TimeUnit.MILLISECONDS), i);
        return new OptimizationResult(X, fX, i);
    }
}
