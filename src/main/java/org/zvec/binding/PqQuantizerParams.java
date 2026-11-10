package org.zvec.binding;

import org.zvec.binding.ZvecNative.zvec_pq_quantizer_params_t;

import java.util.Objects;

/**
 * Product-quantization (PQ) configuration for an IVF vector index, available
 * since zvec v0.8.1.
 *
 * <p>A plain value object: {@link IndexParams#setQuantizerPq(PqQuantizerParams)}
 * copies the values into the native index params, so one instance can be reused
 * and changed afterwards without touching indexes already built from it. The
 * no-argument constructor carries the defaults of
 * {@code ZVEC_PQ_QUANTIZER_PARAMS_DEFAULT} — eight chunks of eight bits, no
 * rotation and no FastScan.
 *
 * <p>PQ quantizes an FP32 vector field indexed as IVF with an L2, IP or COSINE
 * metric. The engine re-validates the combination against the field when the
 * schema is built and rejects the collection with
 * {@link ErrorCode#INVALID_ARGUMENT} when:
 *
 * <ul>
 *   <li>{@code numBits} is neither 4 nor 8;</li>
 *   <li>{@code numChunk} is not positive, exceeds the vector dimension, or —
 *       with 4 bits — does not divide it exactly;</li>
 *   <li>{@code fastScan} is set with 8 bits, FastScan being PQ4-only;</li>
 *   <li>{@code enableRotate} is set while {@code opqIter} or {@code opqPqIter}
 *       is zero.</li>
 * </ul>
 */
public class PqQuantizerParams {

    private int numChunk;
    private int numBits;
    private boolean enableRotate;
    private boolean fastScan;
    private int opqIter;
    private int opqPqIter;

    /** PQ configuration with the engine defaults: 8 chunks x 8 bits. */
    public PqQuantizerParams() {
        this(8, 8, false, false, 5, 4);
    }

    /**
     * Full PQ configuration.
     *
     * @param numChunk     number of sub-vector chunks each vector is split into
     * @param numBits      bits per chunk code word: 4 (PQ4) or 8 (PQ8)
     * @param enableRotate learn a rotation matrix (OPQ) instead of quantizing
     *                     the raw vectors
     * @param fastScan     pack code words for SIMD scanning; requires 4 bits
     * @param opqIter      outer OPQ training iterations
     * @param opqPqIter    inner PQ training iterations per OPQ pass
     */
    public PqQuantizerParams(int numChunk, int numBits, boolean enableRotate,
                             boolean fastScan, int opqIter, int opqPqIter) {
        this.numChunk = numChunk;
        this.numBits = numBits;
        this.enableRotate = enableRotate;
        this.fastScan = fastScan;
        this.opqIter = opqIter;
        this.opqPqIter = opqPqIter;
    }

    /**
     * PQ4: four bits per chunk code word, the compact end of the accuracy /
     * memory trade-off. The vector dimension must be divisible by
     * {@code numChunk}.
     */
    public static PqQuantizerParams pq4(int numChunk) {
        return new PqQuantizerParams(numChunk, 4, false, false, 5, 4);
    }

    /**
     * PQ8: eight bits per chunk code word, the accurate end of the accuracy /
     * memory trade-off.
     */
    public static PqQuantizerParams pq8(int numChunk) {
        return new PqQuantizerParams(numChunk, 8, false, false, 5, 4);
    }

    /**
     * Learn a rotation matrix before quantizing (OPQ), which usually lowers the
     * quantization error at the cost of a longer build. Both iteration counts
     * must stay positive.
     */
    public PqQuantizerParams withRotation(int opqIter, int opqPqIter) {
        this.enableRotate = true;
        this.opqIter = opqIter;
        this.opqPqIter = opqPqIter;
        return this;
    }

    /** Enable or disable FastScan, the SIMD-friendly PQ4 code layout. */
    public PqQuantizerParams withFastScan(boolean fastScan) {
        this.fastScan = fastScan;
        return this;
    }

    public int getNumChunk() {
        return numChunk;
    }

    public void setNumChunk(int numChunk) {
        this.numChunk = numChunk;
    }

    public int getNumBits() {
        return numBits;
    }

    public void setNumBits(int numBits) {
        this.numBits = numBits;
    }

    public boolean getEnableRotate() {
        return enableRotate;
    }

    public void setEnableRotate(boolean enableRotate) {
        this.enableRotate = enableRotate;
    }

    public boolean getFastScan() {
        return fastScan;
    }

    public void setFastScan(boolean fastScan) {
        this.fastScan = fastScan;
    }

    /** Outer OPQ training iterations; the C field is {@code uint32_t}. */
    public int getOpqIter() {
        return opqIter;
    }

    public void setOpqIter(int opqIter) {
        this.opqIter = opqIter;
    }

    /** Inner PQ training iterations per OPQ pass; the C field is {@code uint32_t}. */
    public int getOpqPqIter() {
        return opqPqIter;
    }

    public void setOpqPqIter(int opqPqIter) {
        this.opqPqIter = opqPqIter;
    }

    /**
     * Allocate the native struct and copy these values into it. Package-private:
     * the caller owns the returned pointer and must close it.
     */
    zvec_pq_quantizer_params_t toNative() {
        zvec_pq_quantizer_params_t raw = new zvec_pq_quantizer_params_t();
        raw.num_chunk(numChunk);
        raw.num_bits(numBits);
        raw.enable_rotate(enableRotate);
        raw.fast_scan(fastScan);
        raw.opq_iter(opqIter);
        raw.opq_pq_iter(opqPqIter);
        return raw;
    }

    /** Read a native struct into a new value object. */
    static PqQuantizerParams fromNative(zvec_pq_quantizer_params_t raw) {
        return new PqQuantizerParams(raw.num_chunk(), raw.num_bits(), raw.enable_rotate(),
                raw.fast_scan(), raw.opq_iter(), raw.opq_pq_iter());
    }

    @Override
    public boolean equals(Object other) {
        if (this == other) {
            return true;
        }
        if (!(other instanceof PqQuantizerParams)) {
            return false;
        }
        PqQuantizerParams pq = (PqQuantizerParams) other;
        return numChunk == pq.numChunk
                && numBits == pq.numBits
                && enableRotate == pq.enableRotate
                && fastScan == pq.fastScan
                && opqIter == pq.opqIter
                && opqPqIter == pq.opqPqIter;
    }

    @Override
    public int hashCode() {
        return Objects.hash(numChunk, numBits, enableRotate, fastScan, opqIter, opqPqIter);
    }

    @Override
    public String toString() {
        return "PqQuantizerParams{numChunk=" + numChunk
                + ", numBits=" + numBits
                + ", enableRotate=" + enableRotate
                + ", fastScan=" + fastScan
                + ", opqIter=" + opqIter
                + ", opqPqIter=" + opqPqIter + '}';
    }
}
