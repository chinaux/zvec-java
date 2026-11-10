package org.zvec.binding.examples;

import org.zvec.binding.*;

import java.io.File;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Basic example demonstrating the Zvec Java binding: schema and index setup,
 * insert, vector search, grouped search, fetch, update, delete, stats and the
 * IVF product-quantization parameters.
 *
 * <p>This class is the {@code Main-Class} of the shaded
 * {@code zvec-java-<version>-with-dependencies.jar}, which bundles the natives
 * for every supported platform, so the shortest way to run it is:
 * <pre>
 * mvn package
 * java -jar target/zvec-java-&lt;version&gt;-with-dependencies.jar
 * </pre>
 *
 * <p>To search with your own {@code zvec_c_api} build instead of the bundled
 * one, add {@code -Dzvec.native.path=/path/to/zvec/build/lib} (or export
 * {@code ZVEC_NATIVE_PATH}); see {@link org.zvec.binding.NativeLoader} for the
 * full resolution order.
 */
public class BasicExample {

    private static final Random RAND = new Random(42);
    private static final int DIM = 8;

    public static void main(String[] args) throws Exception {
        // =========================================================================
        // 1. Initialize
        // =========================================================================
        System.out.println("=== Zvec Java Binding - Basic Example ===");

        try (ConfigData config = new ConfigData()) {
            config.setConsoleLog(LogLevel.WARN);
            Zvec.initialize(config);
        }

        System.out.println("Zvec version: " + Zvec.getVersion());

        // =========================================================================
        // 2. Create schema
        // =========================================================================
        File tempDir = Files.createTempDirectory("zvec-java-example-").toFile();
        tempDir.deleteOnExit();
        String dbPath = new File(tempDir, "my_collection").getAbsolutePath();

        CollectionSchema schema = new CollectionSchema("my_collection");

        // Add a vector field with HNSW index
        try (FieldSchema vecField = new FieldSchema("embedding", DataType.VECTOR_FP32, false, DIM)) {
            try (IndexParams hnsw = IndexParams.createHNSW(MetricType.COSINE, 16, 200)) {
                vecField.setIndexParams(hnsw);
            }
            schema.addField(vecField);
        }

        // Add a string field with invert index
        try (FieldSchema titleField = new FieldSchema("title", DataType.STRING, true, 0)) {
            try (IndexParams invert = IndexParams.createInvert(true, false)) {
                titleField.setIndexParams(invert);
            }
            schema.addField(titleField);
        }

        // Add an integer field
        try (FieldSchema yearField = new FieldSchema("year", DataType.INT32, true, 0)) {
            schema.addField(yearField);
        }

        schema.validate();

        // =========================================================================
        // 3. Create and open collection
        // =========================================================================
        // try-with-resources releases the collection even when one of the steps
        // below throws; this is the pattern application code should copy.
        try (Collection collection = Zvec.createAndOpen(dbPath, schema, null)) {
            schema.close();
            run(collection);
        }

        // =========================================================================
        // 12. IVF product quantization parameters (zvec >= v0.8.1)
        // =========================================================================
        ivfProductQuantization();

        Zvec.shutdown();
        System.out.println("\nDone!");
    }

    /** Steps 4 to 11: insert, flush, query, grouped query, fetch, update, delete and stats. */
    private static void run(Collection collection) {

        // =========================================================================
        // 4. Insert documents
        // =========================================================================
        String[] titles = {"Machine Learning Basics", "Deep Learning Advanced", "NLP Fundamentals",
                "Computer Vision", "Reinforcement Learning", "Graph Neural Networks",
                "Transfer Learning", "Attention Mechanism", "Generative Models", "Federated Learning"};
        int[] years = {2020, 2022, 2019, 2021, 2023, 2022, 2021, 2023, 2024, 2023};

        List<Doc> docs = new ArrayList<>();
        for (int i = 0; i < titles.length; i++) {
            Doc doc = new Doc();
            doc.setPK("doc_" + i);
            doc.addVectorFP32Field("embedding", randVec(DIM));
            doc.addStringField("title", titles[i]);
            doc.addInt32Field("year", years[i]);
            docs.add(doc);
        }

        Collection.WriteResult insertResult = collection.insert(docs);
        System.out.println("Inserted: " + insertResult.successCount + " docs");
        Doc.freeDocs(docs);

        // =========================================================================
        // 5. Flush
        // =========================================================================
        collection.flush();

        // =========================================================================
        // 6. Query
        // =========================================================================
        try (VectorQuery query = new VectorQuery()) {
            query.setFieldName("embedding");
            query.setTopK(3);
            query.setQueryVector(randVec(DIM));
            query.setIncludeVector(true);

            List<Doc> results = collection.query(query);
            System.out.println("\nTop-3 search results:");
            for (Doc r : results) {
                System.out.printf("  pk=%s, title=%s, year=%d, score=%.4f%n",
                        r.getPK(),
                        r.getStringField("title"),
                        r.getInt32Field("year"),
                        r.getScore());
            }
            Doc.freeDocs(results);
        }

        // =========================================================================
        // 7. Grouped query (zvec >= v0.8.1)
        // =========================================================================
        // Instead of one flat list, the nearest matches are bucketed by the value
        // of a scalar field: at most groupCount groups come back, each holding at
        // most topkPerGroup documents ordered by similarity — a few documents per
        // year rather than a page full of one year.
        try (GroupByVectorQuery groupQuery = new GroupByVectorQuery()) {
            groupQuery.setFieldName("embedding");
            groupQuery.setGroupByFieldName("year");
            groupQuery.setQueryVector(randVec(DIM));
            groupQuery.setGroupCount(2);
            groupQuery.setTopkPerGroup(2);
            groupQuery.setOutputFields(new String[]{"title", "year"});

            // Each group owns its documents, which stay readable after the query
            // and the collection are closed; release them as a whole.
            List<GroupResult> groups = collection.groupByQuery(groupQuery);
            System.out.println("\nGrouped search over " + groups.size() + " years:");
            for (GroupResult group : groups) {
                System.out.printf("  year=%s (%d docs)%n",
                        group.getGroupByValue(), group.getDocCount());
                for (Doc d : group.getDocs()) {
                    System.out.printf("    pk=%s, title=%s, score=%.4f%n",
                            d.getPK(), d.getStringField("title"), d.getScore());
                }
            }
            GroupResult.freeGroupResults(groups);
        }

        // =========================================================================
        // 8. Fetch by PK (requires forward index on fields)
        // =========================================================================
        try {
            List<String> pks = new ArrayList<>();
            pks.add("doc_0");
            pks.add("doc_5");
            List<Doc> fetched = collection.fetch(pks);
            System.out.println("\nFetched " + fetched.size() + " docs by PK");
            for (Doc d : fetched) {
                System.out.printf("  pk=%s, title=%s%n", d.getPK(), d.getStringField("title"));
            }
            Doc.freeDocs(fetched);
        } catch (ZvecException e) {
            System.out.println("\nFetch note: " + e.getMessage() + " (requires forward index)");
        }

        // =========================================================================
        // 9. Update
        // =========================================================================
        Doc updateDoc = new Doc();
        updateDoc.setPK("doc_0");
        updateDoc.addInt32Field("year", 2025);
        List<Doc> updateDocs = new ArrayList<>();
        updateDocs.add(updateDoc);
        Collection.WriteResult updateResult = collection.update(updateDocs);
        System.out.println("\nUpdated: " + updateResult.successCount + " docs");
        updateDoc.close();

        // =========================================================================
        // 10. Delete
        // =========================================================================
        List<String> deletePks = new ArrayList<>();
        deletePks.add("doc_9");
        Collection.WriteResult deleteResult = collection.delete(deletePks);
        System.out.println("Deleted: " + deleteResult.successCount + " docs");

        // =========================================================================
        // 11. Stats
        // =========================================================================
        CollectionStats stats = collection.getStats();
        System.out.println("\nCollection stats: " + stats);
    }

    /**
     * Configure IVF product quantization, which stores each IVF residual as a
     * product-quantized code and cuts the memory footprint well below IVF
     * RaBitQ's when the dimension is large.
     *
     * <p>{@link IndexParams#createIvfPq} selects {@link QuantizeType#PQ} for
     * you: the C setter copies the chunk layout but leaves the quantize type
     * alone, and PQ settings stored on an index that does not quantize with PQ
     * are ignored silently. The engine validates the chunk layout against the
     * field — its dimension, data type and metric — when the schema is built.
     */
    private static void ivfProductQuantization() {
        try (IndexParams ivfPq = IndexParams.createIvfPq(
                MetricType.COSINE, 100, 20, false, PqQuantizerParams.pq8(8))) {
            PqQuantizerParams pq = ivfPq.getQuantizerPq();
            System.out.printf("%nIVF with PQ: quantize=%s, numChunk=%d, numBits=%d%n",
                    ivfPq.getQuantizeType(), pq.getNumChunk(), pq.getNumBits());
        }

        // OPQ learns a rotation before quantizing, which usually lowers the
        // quantization error at the cost of a longer build. Both iteration counts
        // must stay positive once rotation is enabled.
        try (IndexParams opq = IndexParams.createIVF(MetricType.L2, 64, 20, false)) {
            opq.setQuantizerPq(PqQuantizerParams.pq4(16).withRotation(10, 5));
            PqQuantizerParams pq = opq.getQuantizerPq();
            System.out.printf("IVF with OPQ (learned rotation): numBits=%d, enableRotate=%s, opqIter=%d%n",
                    pq.getNumBits(), pq.getEnableRotate(), pq.getOpqIter());
        }
    }

    private static float[] randVec(int dim) {
        float[] v = new float[dim];
        float norm = 0;
        for (int i = 0; i < dim; i++) {
            v[i] = RAND.nextFloat() * 2 - 1;
            norm += v[i] * v[i];
        }
        norm = (float) Math.sqrt(norm);
        for (int i = 0; i < dim; i++) v[i] /= norm;
        return v;
    }
}
