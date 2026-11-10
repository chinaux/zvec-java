package org.zvec.binding;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Strong-assertion coverage for FieldSchema / CollectionSchema / IndexParams
 * (including the {@code @ByPtrPtr} out-parameter readers), ConfigData, LogConfig
 * and error-code mapping.
 */
class SchemaIndexConfigCoverageTest extends TestSupport {

    // ---------------------------------------------------------------- FieldSchema

    @Test
    void testFieldSchemaAccessors() {
        try (FieldSchema f = new FieldSchema("emb", DataType.VECTOR_FP32, false, 128)) {
            assertEquals("emb", f.getName());
            assertEquals(DataType.VECTOR_FP32, f.getDataType());
            assertEquals(128, f.getDimension());
            assertTrue(f.isVectorField());
            assertTrue(f.isDenseVector());
            assertFalse(f.isSparseVector());
            assertFalse(f.isArrayType());
            assertFalse(f.isNullable());
        }
    }

    @Test
    void testFieldSchemaSetters() {
        try (FieldSchema f = new FieldSchema("a", DataType.VECTOR_FP32, false, 4)) {
            f.setName("renamed");
            assertEquals("renamed", f.getName());
            f.setNullable(true);
            assertTrue(f.isNullable());
            f.setDimension(16);
            assertEquals(16, f.getDimension());
        }
    }

    @Test
    void testFieldSchemaIndexType() {
        try (FieldSchema f = new FieldSchema("emb", DataType.VECTOR_FP32, false, 8)) {
            assertFalse(f.hasIndex());
            try (IndexParams p = IndexParams.createHNSW(MetricType.L2, 8, 100)) {
                f.setIndexParams(p);
            }
            assertTrue(f.hasIndex());
            assertEquals(IndexType.HNSW, f.getIndexType());
        }
    }

    @Test
    void testFieldSchemaInvertIndexFlag() {
        try (FieldSchema f = new FieldSchema("tag", DataType.STRING, true, 0)) {
            try (IndexParams inv = IndexParams.createInvert(true, false)) {
                f.setIndexParams(inv);
            }
            assertTrue(f.hasInvertIndex());
        }
    }

    // ------------------------------------------------------------ CollectionSchema

    @Test
    void testCollectionSchemaFieldLookup() {
        try (CollectionSchema s = new CollectionSchema("look")) {
            try (FieldSchema vec = new FieldSchema("vec", DataType.VECTOR_FP32, false, 4)) {
                s.addField(vec);
            }
            try (FieldSchema cat = new FieldSchema("cat", DataType.STRING, true, 0)) {
                s.addField(cat);
            }
            assertTrue(s.hasField("vec"));
            assertTrue(s.hasField("cat"));

            FieldSchema vec = s.getField("vec");
            assertNotNull(vec);
            assertEquals(DataType.VECTOR_FP32, vec.getDataType());

            assertNotNull(s.getVectorField("vec"));
            assertNotNull(s.getForwardField("cat"));

            List<String> names = s.getAllFieldNames();
            assertEquals(2, names.size());
            assertTrue(names.contains("vec"));
            assertTrue(names.contains("cat"));
        }
    }

    @Test
    void testCollectionSchemaIndexAndDropField() {
        try (CollectionSchema s = new CollectionSchema("idx")) {
            try (FieldSchema vec = new FieldSchema("vec", DataType.VECTOR_FP32, false, 4)) {
                s.addField(vec);
            }
            assertFalse(s.hasIndex("vec"));
            try (IndexParams p = IndexParams.createHNSW(MetricType.L2, 8, 100)) {
                s.addIndex("vec", p);
            }
            assertTrue(s.hasIndex("vec"));
            s.dropIndex("vec");
            assertFalse(s.hasIndex("vec"));

            try (FieldSchema tmp = new FieldSchema("tmp", DataType.INT32, true, 0)) {
                s.addField(tmp);
            }
            assertTrue(s.hasField("tmp"));
            s.dropField("tmp");
            assertFalse(s.hasField("tmp"));
        }
    }

    @Test
    void testCollectionSchemaNameAndSegment() {
        try (CollectionSchema s = new CollectionSchema("orig")) {
            assertEquals("orig", s.getName());
            s.setName("renamed");
            assertEquals("renamed", s.getName());
            s.setMaxDocCountPerSegment(123456L);
            assertEquals(123456L, s.getMaxDocCountPerSegment());
        }
    }

    // ---------------------------------------------------------------- IndexParams

    @Test
    void testIVFParamsRoundtrip() {
        try (IndexParams p = IndexParams.createIVF(MetricType.COSINE, 256, 100, false)) {
            int[] ivf = p.getIVFParams();
            assertEquals(256, ivf[0], "nList");
            assertEquals(100, ivf[1], "nIters");
            assertEquals(0, ivf[2], "useSoar");
        }
        try (IndexParams p = IndexParams.createIVF(MetricType.L2, 64, 50, true)) {
            int[] ivf = p.getIVFParams();
            assertEquals(64, ivf[0]);
            assertEquals(50, ivf[1]);
            assertEquals(1, ivf[2], "useSoar=true");
        }
    }

    @Test
    void testIVFParamsSetter() {
        try (IndexParams p = IndexParams.createIVF(MetricType.L2, 8, 10, false)) {
            p.setIVFParams(512, 200, true);
            int[] ivf = p.getIVFParams();
            assertEquals(512, ivf[0]);
            assertEquals(200, ivf[1]);
            assertEquals(1, ivf[2]);
        }
    }

    @Test
    void testInvertParamsRoundtrip() {
        try (IndexParams p = IndexParams.createInvert(true, false)) {
            boolean[] inv = p.getInvertParams();
            assertTrue(inv[0], "enableRangeOpt");
            assertFalse(inv[1], "enableWildcard");
        }
        try (IndexParams p = IndexParams.createInvert(false, true)) {
            boolean[] inv = p.getInvertParams();
            assertFalse(inv[0]);
            assertTrue(inv[1]);
        }
    }

    @Test
    void testIndexParamsMetricAndQuantize() {
        try (IndexParams p = IndexParams.createFlat(MetricType.L2)) {
            assertEquals(MetricType.L2, p.getMetricType());
            p.setMetricType(MetricType.IP);
            assertEquals(MetricType.IP, p.getMetricType());
        }
        try (IndexParams p = IndexParams.createHNSW(MetricType.COSINE, 16, 200)) {
            p.setQuantizeType(QuantizeType.INT8);
            assertEquals(QuantizeType.INT8, p.getQuantizeType());
            assertEquals(16, p.getHNSWM());
            assertEquals(200, p.getHNSWEfConstruction());
        }
    }

    @Test
    void testFactoryRejectsNullEnumArguments() {
        // A rejected factory call must report which argument was wrong and must
        // not leave the freshly allocated native params behind.
        ZvecException metric = assertThrows(ZvecException.class,
                () -> IndexParams.createHNSW(null, 16, 200));
        assertEquals(ErrorCode.INVALID_ARGUMENT, metric.getErrorCode());
        assertTrue(metric.getMessage().contains("metric"), metric.getMessage());

        ZvecException quantize = assertThrows(ZvecException.class,
                () -> IndexParams.createHNSWQuantized(MetricType.L2, 16, 200, null));
        assertEquals(ErrorCode.INVALID_ARGUMENT, quantize.getErrorCode());
        assertTrue(quantize.getMessage().contains("quantize"), quantize.getMessage());

        assertThrows(ZvecException.class, () -> IndexParams.createFlat(null));
        assertThrows(ZvecException.class, () -> IndexParams.createIVF(null, 8, 10, false));
        assertThrows(ZvecException.class, () -> IndexParams.createIvfRabitq(null, 8, 32, 0));

        ZvecException pq = assertThrows(ZvecException.class,
                () -> IndexParams.createIvfPq(MetricType.L2, 8, 10, false, null));
        assertEquals(ErrorCode.INVALID_ARGUMENT, pq.getErrorCode());
    }

    // ------------------------------------------------- IVF product quantization (zvec >= v0.8.1)

    @Test
    void testPqQuantizerParamsDefaults() {
        // Mirrors ZVEC_PQ_QUANTIZER_PARAMS_DEFAULT: {8, 8, false, false, 5, 4}.
        PqQuantizerParams pq = new PqQuantizerParams();
        assertEquals(8, pq.getNumChunk());
        assertEquals(8, pq.getNumBits());
        assertFalse(pq.getEnableRotate());
        assertFalse(pq.getFastScan());
        assertEquals(5, pq.getOpqIter());
        assertEquals(4, pq.getOpqPqIter());
        assertEquals(new PqQuantizerParams(8, 8, false, false, 5, 4), pq);
        assertEquals(new PqQuantizerParams().hashCode(), pq.hashCode());
        assertTrue(pq.toString().contains("numChunk=8"), pq.toString());
    }

    @Test
    void testPqQuantizerParamsFactories() {
        assertEquals(new PqQuantizerParams(16, 4, false, false, 5, 4), PqQuantizerParams.pq4(16));
        assertEquals(new PqQuantizerParams(3, 8, false, false, 5, 4), PqQuantizerParams.pq8(3));

        PqQuantizerParams rotated = PqQuantizerParams.pq4(8).withRotation(7, 3);
        assertTrue(rotated.getEnableRotate());
        assertEquals(7, rotated.getOpqIter());
        assertEquals(3, rotated.getOpqPqIter());

        PqQuantizerParams fastScan = PqQuantizerParams.pq4(8).withFastScan(true);
        assertTrue(fastScan.getFastScan());
        assertNotEquals(PqQuantizerParams.pq4(8), fastScan);

        PqQuantizerParams mutated = PqQuantizerParams.pq8(4);
        mutated.setNumChunk(6);
        mutated.setNumBits(4);
        mutated.setEnableRotate(true);
        mutated.setFastScan(true);
        mutated.setOpqIter(9);
        mutated.setOpqPqIter(2);
        assertEquals(new PqQuantizerParams(6, 4, true, true, 9, 2), mutated);
    }

    @Test
    void testIvfPqIndexParamsRoundtrip() {
        PqQuantizerParams pq = PqQuantizerParams.pq4(4).withFastScan(true).withRotation(7, 3);
        try (IndexParams p = IndexParams.createIvfPq(MetricType.COSINE, 32, 10, true, pq)) {
            assertEquals(IndexType.IVF, p.getType());
            assertEquals(MetricType.COSINE, p.getMetricType());
            int[] ivf = p.getIVFParams();
            assertEquals(32, ivf[0], "nList");
            assertEquals(10, ivf[1], "nIters");
            assertEquals(1, ivf[2], "useSoar");
            // The factory selects PQ; without it the engine ignores the settings.
            assertEquals(QuantizeType.PQ, p.getQuantizeType());
            assertEquals(pq, p.getQuantizerPq());
        }
    }

    /**
     * Covers the two things the C setter does not do for us: it leaves the
     * quantize type alone, and it copies the struct rather than holding a
     * reference to it.
     */
    @Test
    void testSetQuantizerPqSelectsPqAndCopies() {
        try (IndexParams p = IndexParams.createIVF(MetricType.COSINE, 32, 10, true)) {
            assertEquals(QuantizeType.UNDEFINED, p.getQuantizeType());

            PqQuantizerParams want = PqQuantizerParams.pq8(8);
            p.setQuantizerPq(want);
            assertEquals(QuantizeType.PQ, p.getQuantizeType());
            assertEquals(want, p.getQuantizerPq());

            // The values were copied: changing the Java object afterwards does
            // not reach the native params.
            want.setNumChunk(2);
            assertEquals(8, p.getQuantizerPq().getNumChunk());

            // Five bits per chunk is not a PQ code width: the engine rejects it
            // and the stored configuration must survive untouched.
            PqQuantizerParams rejected = PqQuantizerParams.pq8(8);
            rejected.setNumBits(5);
            ZvecException e = assertThrows(ZvecException.class, () -> p.setQuantizerPq(rejected));
            assertEquals(ErrorCode.INVALID_ARGUMENT, e.getErrorCode());
            assertEquals(PqQuantizerParams.pq8(8), p.getQuantizerPq());
            assertEquals(QuantizeType.PQ, p.getQuantizeType());
        }
    }

    /**
     * Every index type the C setter refuses. HNSW can quantize with PQ, but only
     * through {@link IndexParams#setQuantizeType} with the engine defaults, so it
     * is rejected here too.
     */
    @Test
    void testQuantizerPqRejectsNonIvfParams() {
        IndexParams[] params = {
                IndexParams.createHNSW(MetricType.L2, 16, 200),
                IndexParams.createIvfRabitq(MetricType.L2, 16, 7, 0),
                IndexParams.createFlat(MetricType.L2),
                IndexParams.createInvert(false, false),
        };
        try {
            for (IndexParams p : params) {
                IndexType type = p.getType();
                ZvecException set = assertThrows(ZvecException.class,
                        () -> p.setQuantizerPq(new PqQuantizerParams()), type.name());
                assertEquals(ErrorCode.INVALID_ARGUMENT, set.getErrorCode(), type.name());
                ZvecException get = assertThrows(ZvecException.class, p::getQuantizerPq, type.name());
                assertEquals(ErrorCode.INVALID_ARGUMENT, get.getErrorCode(), type.name());
            }
        } finally {
            for (IndexParams p : params) {
                p.destroy();
            }
        }
    }

    @Test
    void testPqQuantizerParamsRotation() {
        try (IndexParams p = IndexParams.createIVF(MetricType.L2, 16, 10, false)) {
            PqQuantizerParams pq = PqQuantizerParams.pq8(4).withRotation(7, 3);
            p.setQuantizerPq(pq);
            assertEquals(pq, p.getQuantizerPq());

            // Rotation with a zero iteration count is rejected.
            PqQuantizerParams noIter = PqQuantizerParams.pq8(4).withRotation(0, 3);
            ZvecException e = assertThrows(ZvecException.class, () -> p.setQuantizerPq(noIter));
            assertEquals(ErrorCode.INVALID_ARGUMENT, e.getErrorCode());
        }
    }

    // ---------------------------------------------------------------- ConfigData

    @Test
    void testConfigDataRatiosAndThreads() {
        try (ConfigData cfg = new ConfigData()) {
            cfg.setInvertToForwardScanRatio(0.25f);
            assertEquals(0.25f, cfg.getInvertToForwardScanRatio(), 1e-6);
            cfg.setBruteForceByKeysRatio(0.75f);
            assertEquals(0.75f, cfg.getBruteForceByKeysRatio(), 1e-6);
            cfg.setOptimizeThreadCount(3);
            assertEquals(3, cfg.getOptimizeThreadCount());
        }
    }

    @Test
    void testSetLogConfigTransfersOwnership() {
        try (ConfigData cfg = new ConfigData()) {
            LogConfig lc = LogConfig.createFile(LogLevel.WARN, "/tmp/zveclog", "owned", 10, 7);
            cfg.setLogConfig(lc);
            assertEquals(1, cfg.getLogType(), "ZVEC_LOG_TYPE_FILE must be registered");

            // The C config now owns the log config: the wrapper has to be inert,
            // otherwise closing it would free memory the C side still points at.
            assertNull(lc.getHandle());
            lc.close();
            lc.close();
            assertEquals(1, cfg.getLogType());
        }
    }

    // ---------------------------------------------------------------- LogConfig

    @Test
    void testLogConfigFile() {
        try (LogConfig lc = LogConfig.createFile(LogLevel.INFO, "/tmp/zveclog", "app", 10, 7)) {
            assertTrue(lc.isFileType());
            assertEquals("/tmp/zveclog", lc.getDir());
            assertEquals("app", lc.getBasename());
            assertEquals(10, lc.getFileSize());
            assertEquals(7, lc.getOverdueDays());
            assertEquals(LogLevel.INFO, lc.getLevel());
            lc.setLevel(LogLevel.WARN);
            assertEquals(LogLevel.WARN, lc.getLevel());
        }
    }

    @Test
    void testLogConfigConsoleNotFileType() {
        try (LogConfig lc = LogConfig.createConsole(LogLevel.ERROR)) {
            assertFalse(lc.isFileType());
            assertEquals(LogLevel.ERROR, lc.getLevel());
        }
    }

    // ---------------------------------------------------------------- Exceptions

    @Test
    void testOpenNonExistentThrowsWithNativeMessage() {
        ZvecException ex = assertThrows(ZvecException.class,
                () -> Zvec.open("/no/such/zvec/collection/xyz", null));
        String code = ex.getErrorCode().toString();
        assertTrue(ex.getMessage().startsWith(code), ex.getMessage());
        // A bare error code is not actionable: the native layer records why the
        // open failed, and the exception has to carry that along.
        assertTrue(ex.getMessage().length() > code.length(),
                "expected a native message after the error code, got: " + ex.getMessage());
    }

    @Test
    void testErrorCodeMappingAndPredicates() {
        ZvecException ex = new ZvecException(3, "bad arg");
        assertEquals(ErrorCode.INVALID_ARGUMENT, ex.getErrorCode());
        assertTrue(ex.isInvalidArgument());
        assertFalse(ex.isNotFound());
        assertEquals(ErrorCode.NOT_FOUND, ErrorCode.fromCode(1));
        assertEquals(ErrorCode.ALREADY_EXISTS, ErrorCode.fromCode(2));
    }

    @Test
    void testExceptionMessageFormatting() {
        assertEquals("InvalidArgument(3): bad arg", new ZvecException(3, "bad arg").getMessage());
        assertEquals("NotFound(1)", new ZvecException(ErrorCode.NOT_FOUND).getMessage());
        assertEquals("NotFound(1)", new ZvecException(ErrorCode.NOT_FOUND, "").getMessage());
        assertEquals("Unknown(10)", new ZvecException(ErrorCode.UNKNOWN, null).getMessage());
        assertEquals(ErrorCode.INTERNAL_ERROR,
                new ZvecException(ErrorCode.INTERNAL_ERROR, "boom").getErrorCode());
    }
}
