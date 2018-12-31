package com.smagellan.toyml;

import org.apache.commons.lang3.tuple.Triple;
import org.ejml.data.DMatrix;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.RunnerException;
import org.openjdk.jmh.runner.options.Options;
import org.openjdk.jmh.runner.options.OptionsBuilder;
import pl.project13.scala.jmh.extras.profiler.AsyncProfiler;

import java.io.IOException;

public class PerfTest {
    public static void main(String[] args) throws RunnerException {
        Options options = new OptionsBuilder()
                .include(PerfTest.class.getSimpleName())
                .threads(1)
                .forks(1)
                .warmupForks(1)
                .warmupIterations(1)
                .shouldFailOnError(true)
                .shouldDoGC(true)
                .jvmArgs("-server")
                .addProfiler(AsyncProfiler.class, "asyncProfilerDir=/home/vladimir/projects/async-profiler/")
                .build();

        new Runner(options).run();
    }

    @Fork(value = 1, warmups = 2)
    @Benchmark
    @BenchmarkMode(Mode.AverageTime)
    public void benchmark(TrainingExamples examples) {
        new Main().doFit(examples.getParams());
    }

    @State(Scope.Benchmark)
    public static class TrainingExamples {
        private final Triple<DMatrix, DMatrix, Long> params;
        public TrainingExamples() {
            try {
                this.params = Main.loadParams(100);
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }

        public Triple<DMatrix, DMatrix, Long> getParams() {
            return params;
        }
    }
}
