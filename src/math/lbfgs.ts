export interface Optimizable {
    getValue(parameters: number[]): number;
    getGradient(parameters: number[], gradient: number[]): number[];
}

class Vector {
    static copyInto(target: number[], source: number[]): void {
        for (let i = 0; i < source.length; i++) {
            target[i] = source[i];
        }
    }

    static clone(source: number[]): number[] {
        return [...source];
    }

    static rep(dimensions: number[], value: number): number[] {
        const length = dimensions[0];
        return new Array(length).fill(value);
    }

    static sub(a: number[], b: number[]): number[] {
        return a.map((val, i) => val - b[i]);
    }

    static norm2(vector: number[]): number {
        return Math.sqrt(vector.reduce((sum, val) => sum + val * val, 0));
    }

    static diveq(vector: number[], divisor: number): void {
        for (let i = 0; i < vector.length; i++) {
            vector[i] /= divisor;
        }
    }

    static muleq(vector: number[], multiplier: number): void {
        for (let i = 0; i < vector.length; i++) {
            vector[i] *= multiplier;
        }
    }

    static dot(a: number[], b: number[]): number {
        return a.reduce((sum, val, i) => sum + val * b[i], 0);
    }

    static addeq(target: number[], source: number[]): void {
        for (let i = 0; i < target.length; i++) {
            target[i] += source[i];
        }
    }

    static mul(vector: number[], scalar: number): number[] {
        return vector.map(val => val * scalar);
    }
}

// Debug helper function
function checkForNaN(arr: number[], name: string): boolean {
    for (let i = 0; i < arr.length; i++) {
        if (!isFinite(arr[i])) {
            console.error(`NaN/Infinity detected in ${name}[${i}]: ${arr[i]}`);
            return true;
        }
    }
    return false;
}

export function limitedMemoryBFGS(optimizable: Optimizable, parameters: number[]): boolean {
    const lbfgsStart = Date.now();

    let converged = false;
    const maxIterations = 100;
    const tolerance = 0.0001;
    const gradientTolerance = 0.001;
    const epsilon = 0.00001;
    const memorySize = 4;

    const numParameters = parameters.length;

    console.log("Initial parameters:", parameters.slice(0, 5), "...");
    
    // Check initial parameters
    if (checkForNaN(parameters, "initial parameters")) {
        console.error("Initial parameters contain NaN/Infinity");
        return false;
    }

    let gradient = Vector.rep([numParameters], 0.0);
    
    // Get initial gradient and check for NaN
    try {
        optimizable.getGradient(parameters, gradient);
        console.log("Initial gradient:", gradient.slice(0, 5), "...");
        if (checkForNaN(gradient, "initial gradient")) {
            console.error("Initial gradient contains NaN/Infinity");
            return false;
        }
    } catch (error) {
        console.error("Error in getGradient:", error);
        return false;
    }

    let oldGradient = Vector.clone(gradient);
    let oldParameters = Vector.clone(parameters);

    let direction = Vector.clone(gradient);

    // Project direction to the l2 ball
    const directionNorm = Vector.norm2(direction);
    console.log("Direction norm:", directionNorm);
    if (directionNorm === 0) {
        console.error("Direction norm is zero - cannot normalize");
        return false;
    }
    Vector.diveq(direction, directionNorm);

    const parameterChangeBuffer: number[][] = []; // "s"
    const gradientChangeBuffer: number[][] = []; // "y"
    const scaleBuffer: number[] = []; // "rho"

    // Initial step, do a line search in the direction of the gradient
    console.log("Starting line search with direction:", direction.slice(0, 5), "...");
    let scale = backtrackingLineSearch(optimizable, direction, gradient, parameters);
    console.log("Line search returned scale:", scale);

    // "parameters" has now been updated, so get a new value and gradient
    let value;
    try {
        value = optimizable.getValue(parameters);
        console.log("Initial value after line search:", value);
        if (!isFinite(value)) {
            console.error("getValue returned NaN/Infinity:", value);
            return false;
        }
    } catch (error) {
        console.error("Error in getValue:", error);
        return false;
    }
    
    try {
        gradient = optimizable.getGradient(parameters, gradient);
        if (checkForNaN(gradient, "gradient after line search")) {
            return false;
        }
    } catch (error) {
        console.error("Error in getGradient after line search:", error);
        return false;
    }

    let oldValue = value;

    if (scale === 0.0) {
        console.log("Line search can't step in initial direction.");
    }

    for (let iteration = 0; iteration < maxIterations; iteration++) {
        const start = Date.now();

        const currentGradientNorm = Vector.norm2(gradient);
        console.log(`Beginning L-BFGS iteration ${iteration}, v=${value} ||g||=${currentGradientNorm}`);
        
        // Check for NaN in iteration
        if (!isFinite(value)) {
            console.error(`Value is NaN/Infinity at iteration ${iteration}: ${value}`);
            return false;
        }
        if (!isFinite(currentGradientNorm)) {
            console.error(`Gradient norm is NaN/Infinity at iteration ${iteration}: ${currentGradientNorm}`);
            return false;
        }
        if (checkForNaN(parameters, `parameters at iteration ${iteration}`)) {
            return false;
        }

        // Update the buffers with diffs
        if (parameterChangeBuffer.length < memorySize) {
            // If the buffer isn't full yet, add new arrays
            parameterChangeBuffer.unshift(Vector.sub(parameters, oldParameters));
            gradientChangeBuffer.unshift(Vector.sub(gradient, oldGradient));
        } else {
            // Otherwise, reuse the memory from the last array
            const parameterChange = parameterChangeBuffer.pop()!;
            const gradientChange = gradientChangeBuffer.pop()!;
            for (let i = 0; i < numParameters; i++) {
                parameterChange[i] = parameters[i] - oldParameters[i];
                gradientChange[i] = gradient[i] - oldGradient[i];
            }
            parameterChangeBuffer.unshift(parameterChange);
            gradientChangeBuffer.unshift(gradientChange);
        }

        // Save the old values. Gradient will be overwritten, then parameters.
        Vector.copyInto(oldParameters, parameters);
        Vector.copyInto(oldGradient, gradient);

        let sy = 0.0;
        let yy = 0.0;
        for (let i = 0; i < numParameters; i++) {
            sy += parameterChangeBuffer[0][i] * gradientChangeBuffer[0][i];
            yy += gradientChangeBuffer[0][i] * gradientChangeBuffer[0][i];
        }
        const scalingFactor = sy / yy;
        scaleBuffer.unshift(1.0 / sy);

        if (scalingFactor > 0.0) {
            console.log(`Scaling factor greater than zero: ${scalingFactor}`);
        }

        // Renaming the "gradient" array to "direction" -- but it's the same memory.
        Vector.copyInto(direction, gradient);

        // Forward pass, from newest to oldest
        const alpha: number[] = [];
        for (let step = 0; step < parameterChangeBuffer.length; step++) {
            let currentAlpha = 0.0;
            for (let i = 0; i < numParameters; i++) {
                currentAlpha += parameterChangeBuffer[step][i] * direction[i];
            }
            currentAlpha *= scaleBuffer[step];

            alpha.push(currentAlpha);
            for (let i = 0; i < numParameters; i++) {
                direction[i] += gradientChangeBuffer[step][i] * -currentAlpha;
            }
        }

        for (let i = 0; i < numParameters; i++) {
            direction[i] *= scalingFactor;
        }

        // Backward pass, from oldest to newest
        for (let step = parameterChangeBuffer.length - 1; step >= 0; step--) {
            let beta = 0.0;
            for (let i = 0; i < numParameters; i++) {
                beta += gradientChangeBuffer[step][i] * direction[i];
            }
            beta *= scaleBuffer[step];

            const currentAlpha = alpha[step];
            for (let i = 0; i < numParameters; i++) {
                direction[i] += parameterChangeBuffer[step][i] * (currentAlpha - beta);
            }
        }

        // Negate the direction, to maximize rather than minimize
        for (let i = 0; i < numParameters; i++) {
            direction[i] = -direction[i];
        }

        scale = backtrackingLineSearch(optimizable, direction, gradient, parameters);
        if (scale === 0.0) {
            console.log("Cannot step in current direction");
        }

        value = optimizable.getValue(parameters);
        gradient = optimizable.getGradient(parameters, gradient);

        // Test for convergence
        if (2.0 * (value - oldValue) <= tolerance * (Math.abs(value) + Math.abs(oldValue) + epsilon)) {
            console.log(`Value difference below threshold: ${value} - ${oldValue}`);
            const end = Date.now();
            console.log(`Finished iterations ${end - lbfgsStart}`);
            return true;
        }

        const gradientNorm = Vector.norm2(gradient);
        if (gradientNorm < gradientTolerance) {
            console.log(`Gradient norm below threshold: ${gradientNorm}`);
            const end = Date.now();
            console.log(`Finished iterations ${end - lbfgsStart}`);
            return true;
        } else if (gradientNorm === 0.0) {
            console.log("Gradient norm is zero");
            const end = Date.now();
            console.log(`Finished iterations ${end - lbfgsStart}`);
            return true;
        }

        oldValue = value;
    }

    const end = Date.now();
    console.log(`Finished iterations ${end - lbfgsStart}`);
    return true;
}

function backtrackingLineSearch(
    optimizable: Optimizable,
    direction: number[],
    gradient: number[],
    parameters: number[]
): number {
    const numParameters = parameters.length;

    const MAXIMUM_STEP = 100.0;
    const RELATIVE_TOLERANCE = 0.0001;
    const DECREASE_FRACTION = 0.0001;

    let oldScale = 0.0;
    let scale = 1.0;
    let newScale = 0.0;

    let originalValue;
    try {
        originalValue = optimizable.getValue(parameters);
        console.log("Line search - original value:", originalValue);
        if (!isFinite(originalValue)) {
            console.error("Line search - original value is NaN/Infinity:", originalValue);
            return 0.0;
        }
    } catch (error) {
        console.error("Line search - error in getValue:", error);
        return 0.0;
    }
    
    let oldValue = originalValue;

    // Make sure the initial step size isn't too big
    const twoNorm = Vector.norm2(direction);
    if (twoNorm > MAXIMUM_STEP) {
        console.log(`Initial step ${twoNorm} is too big, reducing`);
        Vector.muleq(direction, MAXIMUM_STEP / twoNorm);
    }

    // Get the initial slope of the function of the scale.
    let slope = 0.0;
    for (let i = 0; i < numParameters; i++) {
        slope += gradient[i] * direction[i];
    }

    // Find the minimum acceptable scale value.
    let maxValue = 0.0;
    for (let i = 0; i < numParameters; i++) {
        const v = Math.abs(direction[i] / Math.max(Math.abs(parameters[i]), 1.0));
        if (v > maxValue) {
            maxValue = v;
        }
    }
    const minimumScale = RELATIVE_TOLERANCE / maxValue;

    for (let iteration = 0; iteration < 25; iteration++) {
        for (let i = 0; i < numParameters; i++) {
            parameters[i] += (scale - oldScale) * direction[i];
        }

        if (scale < minimumScale) {
            console.log("Step too small, exiting.");
            return 0.0;
        }

        let value;
        try {
            value = optimizable.getValue(parameters);
            console.log(`Line search iteration ${iteration}: scale=${scale}, value=${value}`);
        } catch (error) {
            console.error(`Line search - error in getValue at iteration ${iteration}:`, error);
            return 0.0;
        }

        if (!isFinite(value)) {
            console.log(`Line search - value is NaN/Infinity: ${value}`);
            newScale = 0.2 * scale;
        } else if (value >= originalValue + DECREASE_FRACTION * scale * slope) {
            console.log(`Line search converged at iteration ${iteration}`);
            return scale;
        } else {
            if (scale === 1.0) {
                // This is only true if this is the first iteration (?)
                newScale = -slope / (2.0 * (value - originalValue - slope));
            } else {
                const x1 = value - originalValue - scale * slope;
                const x2 = oldValue - originalValue - oldScale * slope;
                const oneOverScaleSquared = 1.0 / (scale * scale);
                const oneOverOldScaleSquared = 1.0 / (oldScale * oldScale);
                const oneOverScaleDiff = 1.0 / (scale - oldScale);

                const a = oneOverScaleDiff * (x1 * oneOverScaleSquared - x2 * oneOverOldScaleSquared);
                const b = oneOverScaleDiff * (-x1 * oldScale * oneOverScaleSquared + x2 * scale * oneOverOldScaleSquared);

                if (a === 0.0) {
                    newScale = -slope / (2.0 * b);
                } else {
                    const disc = b * b - 3.0 * a * slope;
                    if (disc < 0.0) {
                        newScale = 0.5 * scale;
                    } else if (b <= 0.0) {
                        newScale = (-b + Math.sqrt(disc)) / (3.0 * a);
                    } else {
                        newScale = -slope / (b + Math.sqrt(disc));
                    }
                }

                if (newScale > 0.5 * scale) {
                    newScale = 0.5 * scale;
                }
            }
        }

        oldValue = value;
        oldScale = scale;
        scale = Math.max(newScale, 0.1 * scale);
    }

    return scale;
}

// Example usage classes
export class QuadraticOptimizable implements Optimizable {
    getValue(parameters: number[]): number {
        const x = parameters[0];
        const y = parameters[1];
        return -3 * x * x - 4 * y * y + 2 * x - 4 * y + 18;
    }

    getGradient(parameters: number[], gradient: number[]): number[] {
        gradient[0] = -6 * parameters[0] + 2;
        gradient[1] = -8 * parameters[1] - 4;
        return gradient;
    }
}

function doubleExp(n: number): number[] {
    const x = new Array(n);
    for (let i = 0; i < n; i++) {
        x[i] = Math.log(Math.random()) * (Math.random() > 0.5 ? 1.0 : -1.0);
    }
    return x;
}

export class RidgeRegression implements Optimizable {
    private covariates: number[][] = [];
    private responses: number[] = [];
    private originalParameters: number[] = doubleExp(100);

    sample(n: number, noise: () => number): void {
        for (let i = 0; i < n; i++) {
            const x = doubleExp(100);
            this.responses.push(Vector.dot(x, this.originalParameters) + noise());
            this.covariates.push(x);
        }
    }

    getValue(parameters: number[]): number {
        let logLikelihood = 0.0;
        for (let i = 0; i < this.covariates.length; i++) {
            const residual = this.responses[i] - Vector.dot(this.covariates[i], parameters);
            logLikelihood += -0.5 * residual * residual;
        }
        return logLikelihood;
    }

    getGradient(parameters: number[], gradient: number[]): number[] {
        // Initialize gradient to zero
        for (let i = 0; i < gradient.length; i++) {
            gradient[i] = 0;
        }
        
        for (let i = 0; i < this.covariates.length; i++) {
            const residual = this.responses[i] - Vector.dot(this.covariates[i], parameters);
            Vector.addeq(gradient, Vector.mul(this.covariates[i], residual));
        }
        return gradient;
    }
}
