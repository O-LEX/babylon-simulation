import { limitedMemoryBFGS, QuadraticOptimizable, RidgeRegression } from './math/lbfgs';

// サンプル1: 二次関数の最適化
function runQuadraticExample() {
    console.log("=== Quadratic Function Optimization Example ===");
    console.log("Optimizing f(x,y) = -3x² - 4y² + 2x - 4y + 18");
    console.log("Expected maximum at x ≈ 0.333, y ≈ -0.5");
    
    const quadratic = new QuadraticOptimizable();
    const parameters = [0.0, 0.0]; // 初期値
    
    console.log(`Initial parameters: [${parameters[0]}, ${parameters[1]}]`);
    console.log(`Initial value: ${quadratic.getValue(parameters)}`);
    
    const converged = limitedMemoryBFGS(quadratic, parameters);
    
    console.log(`Final parameters: [${parameters[0].toFixed(6)}, ${parameters[1].toFixed(6)}]`);
    console.log(`Final value: ${quadratic.getValue(parameters).toFixed(6)}`);
    console.log(`Converged: ${converged}`);
    console.log("");
}

// サンプル2: リッジ回帰の最適化
function runRidgeRegressionExample() {
    console.log("=== Ridge Regression Example ===");
    console.log("Fitting a linear model with L-BFGS optimization");
    
    const ridgeRegression = new RidgeRegression();
    
    // サンプルデータを生成
    const numSamples = 50;
    const noise = () => 0.1 * (Math.random() - 0.5); // ノイズ関数
    ridgeRegression.sample(numSamples, noise);
    
    // 初期パラメータ（ランダム）
    const numParameters = 100;
    const parameters = new Array(numParameters);
    for (let i = 0; i < numParameters; i++) {
        parameters[i] = Math.random() - 0.5;
    }
    
    console.log(`Number of samples: ${numSamples}`);
    console.log(`Number of parameters: ${numParameters}`);
    console.log(`Initial log-likelihood: ${ridgeRegression.getValue(parameters).toFixed(6)}`);
    
    const converged = limitedMemoryBFGS(ridgeRegression, parameters);
    
    console.log(`Final log-likelihood: ${ridgeRegression.getValue(parameters).toFixed(6)}`);
    console.log(`Converged: ${converged}`);
    console.log("");
}

// サンプル3: シンプルな1次元最適化
function runSimple1DExample() {
    console.log("=== Simple 1D Function Optimization Example ===");
    console.log("Optimizing f(x) = -(x-2)² + 5");
    console.log("Expected maximum at x = 2");
    
    // シンプルな1次元関数
    const simple1D = {
        getValue: (parameters: number[]) => {
            const x = parameters[0];
            return -(x - 2) * (x - 2) + 5;
        },
        
        getGradient: (parameters: number[], gradient: number[]) => {
            const x = parameters[0];
            gradient[0] = -2 * (x - 2);
            return gradient;
        }
    };
    
    const parameters = [0.0]; // 初期値
    
    console.log(`Initial parameter: ${parameters[0]}`);
    console.log(`Initial value: ${simple1D.getValue(parameters)}`);
    
    const converged = limitedMemoryBFGS(simple1D, parameters);
    
    console.log(`Final parameter: ${parameters[0].toFixed(6)}`);
    console.log(`Final value: ${simple1D.getValue(parameters).toFixed(6)}`);
    console.log(`Converged: ${converged}`);
    console.log("");
}

// すべてのサンプルを実行
export function runAllLBFGSExamples() {
    console.log("L-BFGS Optimization Examples");
    console.log("============================");
    console.log("");
    
    runSimple1DExample();
    runQuadraticExample();
    runRidgeRegressionExample();
    
    console.log("All examples completed!");
}

// ブラウザ環境で実行する場合
if (typeof window !== 'undefined') {
    // ページが読み込まれたら実行
    window.addEventListener('load', () => {
        runAllLBFGSExamples();
    });
    
    // グローバルに関数を公開
    (window as any).runLBFGSExamples = runAllLBFGSExamples;
}

// Node.js環境で直接実行する場合
if (typeof require !== 'undefined' && require.main === module) {
    runAllLBFGSExamples();
}
