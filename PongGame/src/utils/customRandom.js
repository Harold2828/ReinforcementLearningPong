/**
 * Generates a random number with a Gaussian (normal) distribution.
 * @param {number} mean - The mean (center) of the distribution.
 * @param {number} standardDeviation - The standard deviation (spread) of the distribution.
 * @returns {number} A random number with a Gaussian distribution.
 */
export function randomNormal(mean = 0, standardDeviation = 1) {
    let uniformRandom1 = 0, uniformRandom2 = 0;
    while (uniformRandom1 === 0) uniformRandom1 = Math.random(); // Avoid 0
    while (uniformRandom2 === 0) uniformRandom2 = Math.random(); // Avoid 0
    const gaussianRepresentation = Math.sqrt(-2.0 * Math.log(uniformRandom1)) * Math.cos(2.0 * Math.PI * uniformRandom2);
    return gaussianRepresentation * standardDeviation + mean;
}