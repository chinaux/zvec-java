package org.zvec.binding;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Strong-assertion coverage for DML/DQL: query top-K count / ordering / PK hit,
 * output-field retrieval, filtering, update read-back and delete verification.
 * Follows the project's strong-assertion testing convention (no "does-not-throw"
 * only checks; results are validated).
 */
class CollectionQueryCoverageTest extends TestSupport {

    private List<Doc> queryVec(Collection coll, float[] q, int topK, String[] outputFields) {
        try (VectorQuery vq = new VectorQuery()) {
            vq.setFieldName("vec");
            vq.setTopK(topK);
            vq.setQueryVector(q);
            if (outputFields != null) {
                vq.setOutputFields(outputFields);
            }
            return coll.query(vq);
        }
    }

    @Test
    void testQueryTopKAndOrdering(@TempDir Path dir) {
        try (Collection coll = openIndexed(dir, "q_order")) {
            insertGraded(coll, 8);
            coll.flush();

            List<Doc> res = queryVec(coll, vecOf(3), 3, null);
            try {
                assertEquals(3, res.size(), "top-K count");
                // Exact-match vector must be the nearest neighbour.
                assertEquals("pk_3", res.get(0).getPK(), "exact match should rank first");

                // Scores must be monotonic (sorted by distance/similarity).
                boolean asc = true, desc = true;
                for (int i = 1; i < res.size(); i++) {
                    float prev = res.get(i - 1).getScore();
                    float cur = res.get(i).getScore();
                    if (prev > cur) asc = false;
                    if (prev < cur) desc = false;
                }
                assertTrue(asc || desc, "results must be sorted by score");
            } finally {
                Doc.freeDocs(res);
            }
        }
    }

    @Test
    void testQueryReturnsScalarFields(@TempDir Path dir) {
        try (Collection coll = openIndexed(dir, "q_scalar")) {
            insertGraded(coll, 8);
            coll.flush();

            List<Doc> res = queryVec(coll, vecOf(2), 1, new String[]{"cat", "num"});
            try {
                assertEquals(1, res.size());
                Doc r = res.get(0);
                assertEquals("pk_2", r.getPK());
                assertEquals("c0", r.getStringField("cat")); // 2 % 2 == 0
                assertEquals(2, r.getInt32Field("num"));
            } finally {
                Doc.freeDocs(res);
            }
        }
    }

    @Test
    void testQueryWithFilter(@TempDir Path dir) {
        try (Collection coll = openIndexed(dir, "q_filter")) {
            insertGraded(coll, 8);
            coll.flush();

            try (VectorQuery vq = new VectorQuery()) {
                vq.setFieldName("vec");
                vq.setTopK(8);
                vq.setQueryVector(vecOf(0));
                vq.setFilter("num >= 4");
                vq.setOutputFields(new String[]{"num"});
                List<Doc> res = coll.query(vq);
                try {
                    assertFalse(res.isEmpty(), "filter should still match docs");
                    for (Doc r : res) {
                        assertTrue(r.getInt32Field("num") >= 4,
                                "filtered result num must satisfy num>=4 but was " + r.getInt32Field("num"));
                    }
                } finally {
                    Doc.freeDocs(res);
                }
            }
        }
    }

    @Test
    void testUpdateReadBackViaQuery(@TempDir Path dir) {
        try (Collection coll = openIndexed(dir, "q_update")) {
            insertGraded(coll, 8);
            coll.flush();

            // Full-doc update of pk_2.
            Doc upd = new Doc();
            upd.setPK("pk_2");
            upd.addVectorFP32Field("vec", vecOf(2));
            upd.addStringField("cat", "updated");
            upd.addInt32Field("num", 999);
            List<Doc> batch = new ArrayList<>();
            batch.add(upd);
            Collection.WriteResult wr = coll.update(batch);
            Doc.freeDocs(batch);
            assertEquals(1, wr.successCount, "update should affect exactly 1 doc");
            coll.flush();

            List<Doc> res = queryVec(coll, vecOf(2), 1, new String[]{"cat", "num"});
            try {
                assertEquals(1, res.size());
                assertEquals("pk_2", res.get(0).getPK());
                assertEquals("updated", res.get(0).getStringField("cat"), "cat must reflect update");
                assertEquals(999, res.get(0).getInt32Field("num"), "num must reflect update");
            } finally {
                Doc.freeDocs(res);
            }
        }
    }

    @Test
    void testDeleteRemovesFromQuery(@TempDir Path dir) {
        try (Collection coll = openIndexed(dir, "q_delete")) {
            insertGraded(coll, 8);
            coll.flush();

            // pk_5 present before delete.
            List<Doc> before = queryVec(coll, vecOf(5), 1, null);
            try {
                assertEquals("pk_5", before.get(0).getPK());
            } finally {
                Doc.freeDocs(before);
            }

            Collection.WriteResult wr = coll.delete(java.util.Collections.singletonList("pk_5"));
            assertEquals(1, wr.successCount, "delete should remove exactly 1 doc");
            coll.flush();

            // pk_5 must no longer appear anywhere in the result set.
            List<Doc> after = queryVec(coll, vecOf(5), 8, null);
            try {
                for (Doc r : after) {
                    assertNotEquals("pk_5", r.getPK(), "deleted pk_5 must not appear");
                }
            } finally {
                Doc.freeDocs(after);
            }
        }
    }

    @Test
    void testDeleteByFilterRemovesMatching(@TempDir Path dir) {
        try (Collection coll = openIndexed(dir, "q_delfilter")) {
            insertGraded(coll, 8); // even -> c0, odd -> c1
            coll.flush();

            coll.deleteByFilter("cat = 'c1'");
            coll.flush();

            try (VectorQuery vq = new VectorQuery()) {
                vq.setFieldName("vec");
                vq.setTopK(8);
                vq.setQueryVector(vecOf(0));
                vq.setOutputFields(new String[]{"cat"});
                List<Doc> res = coll.query(vq);
                try {
                    assertFalse(res.isEmpty());
                    for (Doc r : res) {
                        assertEquals("c0", r.getStringField("cat"),
                                "all c1 docs should have been deleted");
                    }
                } finally {
                    Doc.freeDocs(res);
                }
            }
        }
    }

    @Test
    void testClosedCollectionRejectsOperations(@TempDir Path dir) {
        Collection coll = openIndexed(dir, "closed_guard");
        assertTrue(coll.isOpen());
        coll.close();
        assertFalse(coll.isOpen(), "isOpen() must report a closed collection");

        // A released handle has to surface as an exception; letting it reach the
        // native layer turns a programming error into a JVM crash.
        ZvecException flush = assertThrows(ZvecException.class, coll::flush);
        assertEquals(ErrorCode.FAILED_PRECONDITION, flush.getErrorCode());
        assertTrue(flush.getMessage().contains("closed"), flush.getMessage());

        assertThrows(ZvecException.class, coll::optimize);
        assertThrows(ZvecException.class, coll::getStats);
        assertThrows(ZvecException.class, coll::getSchema);
        assertThrows(ZvecException.class, () -> coll.deleteByFilter("num > 0"));
        assertThrows(ZvecException.class, () -> coll.createIterator(null));
        try (VectorQuery vq = new VectorQuery()) {
            vq.setFieldName("vec");
            vq.setTopK(1);
            vq.setQueryVector(vecOf(1));
            assertThrows(ZvecException.class, () -> coll.query(vq));
        }

        // close() and destroy() stay idempotent once the handle is gone.
        coll.close();
        coll.destroy();
    }

    @Test
    void testQueryWithoutMatchesReturnsEmptyList(@TempDir Path dir) {
        try (Collection coll = openIndexed(dir, "q_empty")) {
            insertGraded(coll, 4);
            coll.flush();

            // A filter that matches nothing makes the C API allocate a
            // zero-element result array; the wrapper still has to release it.
            try (VectorQuery vq = new VectorQuery()) {
                vq.setFieldName("vec");
                vq.setTopK(4);
                vq.setQueryVector(vecOf(0));
                vq.setFilter("num >= 1000");
                List<Doc> res = coll.query(vq);
                assertNotNull(res);
                assertTrue(res.isEmpty(), "no document matches the filter");
            }

            List<Doc> missing = coll.fetch(Collections.singletonList("does_not_exist"));
            assertNotNull(missing);
            assertTrue(missing.isEmpty(), "an unknown PK must not produce a document");
        }
    }

    @Test
    void testBatchWriteRejectsClosedDocument(@TempDir Path dir) {
        try (Collection coll = openIndexed(dir, "closed_doc")) {
            Doc closed = new Doc();
            closed.setPK("gone");
            closed.addVectorFP32Field("vec", vecOf(0));
            closed.close();

            // The C API dereferences every element of a write batch without a
            // NULL check, so a closed document must be rejected in Java rather
            // than handed down as a NULL pointer that crashes the JVM.
            ZvecException insert = assertThrows(ZvecException.class,
                    () -> coll.insert(Collections.singletonList(closed)));
            assertEquals(ErrorCode.FAILED_PRECONDITION, insert.getErrorCode());
            assertTrue(insert.getMessage().contains("closed"), insert.getMessage());

            assertThrows(ZvecException.class, () -> coll.upsert(Collections.singletonList(closed)));
            assertThrows(ZvecException.class, () -> coll.update(Collections.singletonList(closed)));

            List<Doc> withNull = new ArrayList<>();
            withNull.add(null);
            ZvecException nullDoc = assertThrows(ZvecException.class, () -> coll.insert(withNull));
            assertEquals(ErrorCode.INVALID_ARGUMENT, nullDoc.getErrorCode());
        }
    }

    @Test
    void testNullOutputFieldsReturnsEveryField(@TempDir Path dir) {
        try (Collection coll = openIndexed(dir, "null_fields")) {
            insertGraded(coll, 4);
            coll.flush();

            // A null list means "every field"; it used to throw a
            // NullPointerException before the array was even built.
            try (VectorQuery vq = new VectorQuery()) {
                vq.setFieldName("vec");
                vq.setTopK(1);
                vq.setQueryVector(vecOf(1));
                vq.setOutputFields(null);
                List<Doc> res = coll.query(vq);
                try {
                    assertEquals(1, res.size());
                    Doc hit = res.get(0);
                    assertEquals("pk_1", hit.getPK());
                    assertEquals("c1", hit.getStringField("cat"));
                    assertEquals(1, hit.getInt32Field("num"));
                } finally {
                    Doc.freeDocs(res);
                }
            }
        }
    }

    // ---------------------------------------------------------------- Grouped search (zvec >= v0.8.1)

    /**
     * Build a Flat/L2 collection whose documents carry a {@code group_id} that
     * changes every three documents, so a grouped search has something to
     * bucket. Vectors are {@code [id, 0, 0, 0]}, which makes the L2 distance to
     * the origin exactly {@code id*id} and the expected ordering easy to assert.
     */
    private Collection openGroupByCollection(Path dir, String name, int docCount) {
        CollectionSchema schema = new CollectionSchema(name);
        try (FieldSchema id = new FieldSchema("id", DataType.INT64, false, 0)) {
            schema.addField(id);
        }
        try (FieldSchema groupId = new FieldSchema("group_id", DataType.INT64, false, 0)) {
            schema.addField(groupId);
        }
        try (FieldSchema embedding = new FieldSchema("embedding", DataType.VECTOR_FP32, false, 4)) {
            try (IndexParams flat = IndexParams.createFlat(MetricType.L2)) {
                embedding.setIndexParams(flat);
            }
            schema.addField(embedding);
        }
        Collection coll = Zvec.createAndOpen(dir.resolve("coll").toString(), schema, null);
        schema.close();
        if (docCount == 0) {
            return coll;
        }
        List<Doc> docs = new ArrayList<>(docCount);
        for (int id = 0; id < docCount; id++) {
            Doc doc = new Doc();
            doc.setPK("doc_" + id);
            doc.addInt64Field("id", id);
            doc.addInt64Field("group_id", id / 3);
            doc.addVectorFP32Field("embedding", new float[]{id, 0f, 0f, 0f});
            docs.add(doc);
        }
        Collection.WriteResult wr = coll.insert(docs);
        Doc.freeDocs(docs);
        assertEquals(docCount, wr.successCount, "all documents should insert");
        coll.flush();
        return coll;
    }

    /** A grouped query over the origin returning only the {@code id} field. */
    private GroupByVectorQuery newGroupByQuery(int groupCount, int topkPerGroup) {
        GroupByVectorQuery query = new GroupByVectorQuery();
        query.setFieldName("embedding");
        query.setGroupByFieldName("group_id");
        query.setQueryVector(new float[]{0f, 0f, 0f, 0f});
        query.setGroupCount(groupCount);
        query.setTopkPerGroup(topkPerGroup);
        query.setOutputFields(new String[]{"id"});
        // setFlatParams takes ownership of the params.
        query.setFlatParams(new FlatQueryParams(false, 10.0f));
        return query;
    }

    @Test
    void testGroupByQueryOnEmptyCollectionReturnsNoGroups(@TempDir Path dir) {
        try (Collection coll = openGroupByCollection(dir, "gb_empty", 0);
             GroupByVectorQuery query = newGroupByQuery(2, 3)) {
            List<GroupResult> groups = coll.groupByQuery(query);
            assertTrue(groups.isEmpty(), "an empty collection has no group");
        }
    }

    @Test
    void testGroupByQueryBucketsResults(@TempDir Path dir) {
        try (Collection coll = openGroupByCollection(dir, "gb_buckets", 9);
             GroupByVectorQuery query = newGroupByQuery(2, 2)) {
            List<GroupResult> groups = coll.groupByQuery(query);
            try {
                assertEquals(2, groups.size(), "the two nearest groups");
                String[] wantValues = {"0", "1"};
                for (int g = 0; g < groups.size(); g++) {
                    GroupResult group = groups.get(g);
                    assertEquals(wantValues[g], group.getGroupByValue());
                    assertEquals(2, group.getDocCount());
                    List<Doc> docs = group.getDocs();
                    for (int pos = 0; pos < docs.size(); pos++) {
                        long id = g * 3L + pos;
                        Doc doc = docs.get(pos);
                        assertEquals("doc_" + id, doc.getPK());
                        assertEquals(id, doc.getInt64Field("id"));
                        // Flat/L2 distance from [id,0,0,0] to the origin is id*id.
                        assertEquals(id * id, doc.getScore(), 1e-3f);
                        assertFalse(doc.hasField("embedding"), "only the requested field comes back");
                        assertFalse(doc.hasField("group_id"), "only the requested field comes back");
                    }
                }
            } finally {
                GroupResult.freeGroupResults(groups);
            }
        }
    }

    @Test
    void testGroupByQueryHonoursFilterAndIncludeVector(@TempDir Path dir) {
        try (Collection coll = openGroupByCollection(dir, "gb_filter", 9);
             GroupByVectorQuery query = newGroupByQuery(3, 1)) {
            query.setFilter("id >= 3");
            query.setIncludeVector(true);
            List<GroupResult> groups = coll.groupByQuery(query);
            try {
                // Group 0 only holds ids 0..2, so the filter drops it.
                assertEquals(2, groups.size());
                for (GroupResult group : groups) {
                    assertEquals(1, group.getDocCount());
                    long groupId = Long.parseLong(group.getGroupByValue());
                    assertTrue(groupId >= 1, "group " + groupId + " survived a filter of id >= 3");
                    long id = groupId * 3;
                    Doc doc = group.getDocs().get(0);
                    assertEquals("doc_" + id, doc.getPK());
                    assertEquals(id, doc.getInt64Field("id"));
                    assertArrayEquals(new float[]{id, 0f, 0f, 0f},
                            doc.getVectorFP32Field("embedding"), 0f);
                }
            } finally {
                GroupResult.freeGroupResults(groups);
            }
        }
    }

    @Test
    void testGroupByQueryWithoutMatchesReturnsNoGroups(@TempDir Path dir) {
        try (Collection coll = openGroupByCollection(dir, "gb_none", 9);
             GroupByVectorQuery query = newGroupByQuery(3, 1)) {
            query.setFilter("id < 0");
            assertTrue(coll.groupByQuery(query).isEmpty(), "no document matches the filter");
        }
    }

    @Test
    void testGroupByQueryRejectsInvalidArguments(@TempDir Path dir) {
        try (Collection coll = openGroupByCollection(dir, "gb_invalid", 9)) {
            try (GroupByVectorQuery zeroGroups = newGroupByQuery(0, 1)) {
                ZvecException e = assertThrows(ZvecException.class, () -> coll.groupByQuery(zeroGroups));
                assertEquals(ErrorCode.INVALID_ARGUMENT, e.getErrorCode());
            }
            try (GroupByVectorQuery missingField = newGroupByQuery(2, 1)) {
                missingField.setGroupByFieldName("no_such_field");
                ZvecException e = assertThrows(ZvecException.class, () -> coll.groupByQuery(missingField));
                assertEquals(ErrorCode.INVALID_ARGUMENT, e.getErrorCode());
            }
            // A null query would reach the C API as a NULL pointer; the wrapper
            // rejects it instead of letting the engine dereference it.
            ZvecException nullQuery = assertThrows(ZvecException.class, () -> coll.groupByQuery(null));
            assertEquals(ErrorCode.INVALID_ARGUMENT, nullQuery.getErrorCode());
        }
    }

    /**
     * The C result array owns the grouping values, every group's document array
     * and the documents themselves, and {@code zvec_group_results_free} is the
     * only way to release it. The wrapper moves the documents into {@link Doc}s
     * and clears each group's array before handing the shell back, so the
     * documents survive the collection and are not destroyed twice.
     */
    @Test
    void testGroupByQueryResultsOutliveTheCollection(@TempDir Path dir) {
        Collection coll = openGroupByCollection(dir, "gb_outlive", 6);
        List<GroupResult> groups;
        try (GroupByVectorQuery query = newGroupByQuery(2, 3)) {
            groups = coll.groupByQuery(query);
        }
        coll.close();
        try {
            assertFalse(coll.isOpen());
            assertEquals(2, groups.size());
            assertEquals("0", groups.get(0).getGroupByValue());
            assertEquals(3, groups.get(0).getDocCount());
            assertEquals("doc_0", groups.get(0).getDocs().get(0).getPK());
            assertEquals("1", groups.get(1).getGroupByValue());
            assertEquals("doc_4", groups.get(1).getDocs().get(1).getPK());
            assertEquals(4L, groups.get(1).getDocs().get(1).getInt64Field("id"));
        } finally {
            GroupResult.freeGroupResults(groups);
        }
        // Releasing twice must be a no-op, not a double free.
        GroupResult.freeGroupResults(groups);
    }
}
